# Orchestrator Specification (CPU Batch Job)

The orchestrator runs as a CPU Batch job, separate from Cloud Run. This provides:
- State persistence across browser disconnects
- No Cloud Run timeout concerns (60 min limit)
- Parallel runs per user
- Cost efficiency (only pay when running)

### 10.1 Orchestrator Entrypoint

```python
# orchestrator/main.py
"""Entrypoint for the CPU Batch job orchestrator."""

import os
import sys
import asyncio
import logging

from orchestrator.config import settings
from orchestrator.services.database import database
from orchestrator.services.email import email_service
from orchestrator.agents.orchestrator import OrchestratorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main orchestrator workflow."""
    # Get run parameters from environment
    run_id = os.environ.get("RUN_ID")
    user_email = os.environ.get("USER_EMAIL")
    env_control_strategy = os.environ.get("CONTROL_STRATEGY")
    
    if not run_id:
        logger.error("RUN_ID environment variable not set")
        sys.exit(1)
    
    logger.info(f"Starting orchestrator for run {run_id}")
    
    try:
        # Connect to database
        await database.connect()
        email_service.initialize()
        
        # Load run configuration from database
        run = await database.get_run(run_id)
        if not run:
            logger.error(f"Run {run_id} not found in database")
            sys.exit(1)
        
        query = run["query"]
        config = run["config"]

        if env_control_strategy:
            # Allow Batch job environment to override config default
            config["control_strategy"] = env_control_strategy
        
        # Update status to running
        await database.update_run(
            run_id=run_id,
            status="running",
            current_phase="query_analysis",
        )
        
        # Initialize and run orchestrator agent
        agent = OrchestratorAgent(
            run_id=run_id,
            query=query,
            user_email=user_email,
            config=config,
        )
        
        result = await agent.run()
        
        # Send completion email
        if result.success:
            await email_service.send_run_completed(
                recipient_email=user_email,
                run_id=run_id,
                query=query,
                grounding_score=result.final_score,
            )
            logger.info(f"Run {run_id} completed successfully with score {result.final_score}")
        else:
            await email_service.send_run_failed(
                recipient_email=user_email,
                run_id=run_id,
                query=query,
                error_message=result.error_message,
            )
            logger.error(f"Run {run_id} failed: {result.error_message}")
        
    except Exception as e:
        logger.exception(f"Orchestrator failed for run {run_id}")
        
        # Update database with error
        await database.update_run(
            run_id=run_id,
            status="failed",
            error_message=str(e),
        )
        
        # Send failure email
        try:
            await email_service.send_run_failed(
                recipient_email=user_email,
                run_id=run_id,
                query=run.get("query", ""),
                error_message=str(e),
            )
        except Exception:
            pass  # Don't fail on email error
        
        sys.exit(1)
        
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.2 Orchestrator Agent

```python
# orchestrator/agents/orchestrator.py
"""Main orchestrator agent that runs the iterative workflow."""

import logging
from dataclasses import dataclass
from typing import Optional

from orchestrator.services.database import database
from orchestrator.services.batch import gpu_batch_client
from orchestrator.services.gcs import gcs_service
from orchestrator.agents.query_understanding import QueryUnderstandingAgent
from orchestrator.agents.prompt_generation import PromptGenerationAgent
from orchestrator.agents.grounding_evaluation import GroundingEvaluationAgent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result of orchestrator execution."""
    success: bool
    final_score: Optional[int] = None
    error_message: Optional[str] = None


class OrchestratorAgent:
    """
    Main orchestrator that runs the iterative HAYSTACK workflow.
    
    Runs in a CPU Batch job and submits GPU Batch jobs for STACK inference.
    """
    
    def __init__(
        self,
        run_id: str,
        query: str,
        user_email: str,
        config: dict,
    ):
        self.run_id = run_id
        self.query = query
        self.user_email = user_email
        self.config = config
        
        self.max_iterations = config.get("max_iterations", 5)
        self.score_threshold = config.get("score_threshold", 7)
        self.control_strategy = config.get("control_strategy", "synthetic_control")
        
        # Initialize subagents
        self.query_agent = QueryUnderstandingAgent()
        self.prompt_agent = PromptGenerationAgent()
        self.evaluation_agent = GroundingEvaluationAgent()
    
    async def run(self) -> OrchestratorResult:
        """Execute the full iterative workflow."""
        try:
            # Step 1: Query understanding
            logger.info(f"[{self.run_id}] Starting query analysis")
            await self._update_phase("query_analysis")
            
            structured_query = await self.query_agent.analyze(self.query)
            structured_query.control_strategy = self.control_strategy

            # Resolve control strategy feasibility
            if self.control_strategy == "synthetic_control":
                control_info = await self.prompt_agent.find_matched_controls(structured_query)
                structured_query.control_cells_available = bool(control_info)
                structured_query.control_cell_info = control_info
                if not control_info:
                    structured_query.control_strategy_fallback = "query_as_control"
                    self.control_strategy = "query_as_control"
            
            await database.update_run(
                run_id=self.run_id,
                control_strategy=structured_query.control_strategy,
                control_strategy_effective=self.control_strategy,
                control_cells_available=structured_query.control_cells_available,
            )
            
            # Iteration loop
            scores = []
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"[{self.run_id}] Starting iteration {iteration}")
                
                # Check for cancellation
                if await self._is_cancelled():
                    logger.info(f"[{self.run_id}] Run cancelled")
                    return OrchestratorResult(success=False, error_message="Cancelled by user")
                
                # Step 2: Prompt generation
                await self._update_phase("prompt_generation", iteration)
                prompt_cells = await self.prompt_agent.generate(
                    structured_query=structured_query,
                    iteration=iteration,
                    previous_scores=scores,
                )
                
                # Step 3: STACK inference (GPU Batch job)
                await self._update_phase("inference", iteration)
                predictions, control_predictions = await self._run_inference(
                    iteration, prompt_cells, structured_query
                )
                
                # Check for cancellation after long inference
                if await self._is_cancelled():
                    return OrchestratorResult(success=False, error_message="Cancelled by user")
                
                # Step 4: Grounding evaluation
                await self._update_phase("evaluation", iteration)
                eval_result = await self.evaluation_agent.evaluate(
                    predictions=predictions,
                    control_predictions=control_predictions,
                    query=structured_query,
                )
                
                score = eval_result.score
                scores.append(score)
                
                # Update database with score
                await database.update_run(
                    run_id=self.run_id,
                    grounding_scores=scores,
                    current_iteration=iteration,
                )
                
                logger.info(f"[{self.run_id}] Iteration {iteration} score: {score}")
                
                # Check convergence
                if score >= self.score_threshold:
                    logger.info(f"[{self.run_id}] Converged with score {score}")
                    break
            
            # Step 5: Output generation
            await self._update_phase("output_generation")
            output_paths = await self._generate_outputs(predictions, scores)
            
            # Update database with completion
            final_score = scores[-1] if scores else 0
            await database.update_run(
                run_id=self.run_id,
                status="completed",
                final_score=final_score,
                output_anndata_path=output_paths["anndata"],
                output_report_path=output_paths["report"],
                output_log_path=output_paths["log"],
            )
            
            return OrchestratorResult(success=True, final_score=final_score)
            
        except Exception as e:
            logger.exception(f"[{self.run_id}] Orchestrator error")
            await database.update_run(
                run_id=self.run_id,
                status="failed",
                error_message=str(e),
            )
            return OrchestratorResult(success=False, error_message=str(e))
    
    async def _is_cancelled(self) -> bool:
        """Check if the run has been cancelled."""
        run = await database.get_run(self.run_id)
        return run.get("status") == "cancelled"
    
    async def _update_phase(self, phase: str, iteration: int = None):
        """Update the current phase in the database."""
        update = {"current_phase": phase}
        if iteration:
            update["current_iteration"] = iteration
        await database.update_run(run_id=self.run_id, **update)
    
    async def _run_inference(self, iteration: int, prompt_cells, structured_query) -> tuple:
        """Submit GPU Batch job(s) for STACK inference."""
        # Write inputs to GCS
        gcs_prefix = f"batch-io/{self.run_id}/iter_{iteration}"
        await gcs_service.write_anndata(f"{gcs_prefix}/prompt.h5ad", prompt_cells)
        
        # Submit GPU job and wait for completion
        predictions_path = await gpu_batch_client.run_inference(
            run_id=self.run_id,
            iteration=iteration,
            prompt_path=f"{gcs_prefix}/prompt.h5ad",
            output_path=f"{gcs_prefix}/predictions.h5ad",
        )

        control_predictions_path = None
        if self.control_strategy == "synthetic_control":
            await gcs_service.write_anndata(
                f"{gcs_prefix}/control_prompt.h5ad",
                structured_query.control_cell_info["anndata"],
            )
            control_predictions_path = await gpu_batch_client.run_inference(
                run_id=self.run_id,
                iteration=iteration,
                prompt_path=f"{gcs_prefix}/control_prompt.h5ad",
                output_path=f"{gcs_prefix}/control_predictions.h5ad",
            )
        
        # Read predictions from GCS
        predictions = await gcs_service.read_anndata(predictions_path)
        control_predictions = (
            await gcs_service.read_anndata(control_predictions_path)
            if control_predictions_path
            else None
        )
        return predictions, control_predictions
    
    async def _generate_outputs(self, predictions, scores) -> dict:
        """Generate final output files."""
        # Implementation details...
        pass
```

### 10.3 Email Notification Service (Orchestrator)

```python
# orchestrator/services/email.py
"""Email notification service using SendGrid."""

import logging
from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from orchestrator.config import settings

logger = logging.getLogger(__name__)
        
        subject = f"HAYSTACK run complete (score: {grounding_score}/10)"
        
        html_content = f"""
        <h2>Your HAYSTACK run is complete!</h2>
        
        <p><strong>Query:</strong> {query[:200]}{'...' if len(query) > 200 else ''}</p>
        
        <p><strong>Grounding Score:</strong> {grounding_score}/10</p>
        
        <p>
            <a href="{results_url}" style="
                display: inline-block;
                padding: 12px 24px;
                background-color: #2563eb;
                color: white;
                text-decoration: none;
                border-radius: 6px;
            ">View Results</a>
        </p>
        
        <p style="color: #6b7280; font-size: 14px;">
            This is an automated message from HAYSTACK.
        </p>
        """
        
        return await self._send_email(recipient_email, subject, html_content)
    
    async def send_run_failed(
        self,
        recipient_email: str,
        run_id: str,
        query: str,
        error_message: str,
    ) -> bool:
        """Send email notification when a run fails."""
        if not self.client:
            return False
        
        results_url = f"{self.base_url}/runs/{run_id}"
        
        subject = "HAYSTACK run failed"
        
        html_content = f"""
        <h2>Your HAYSTACK run encountered an error</h2>
        
        <p><strong>Query:</strong> {query[:200]}{'...' if len(query) > 200 else ''}</p>
        
        <p><strong>Error:</strong> {error_message}</p>
        
        <p>
            <a href="{results_url}">View run details</a>
        </p>
        
        <p style="color: #6b7280; font-size: 14px;">
            Please try again or contact support if the issue persists.
        </p>
        """
        
        return await self._send_email(recipient_email, subject, html_content)
    
    async def send_run_cancelled(
        self,
        recipient_email: str,
        run_id: str,
        query: str,
    ) -> bool:
        """Send email notification when a run is cancelled."""
        if not self.client:
            return False
        
        subject = "HAYSTACK run cancelled"
        
        html_content = f"""
        <h2>Your HAYSTACK run was cancelled</h2>
        
        <p><strong>Query:</strong> {query[:200]}{'...' if len(query) > 200 else ''}</p>
        
        <p style="color: #6b7280; font-size: 14px;">
            You can start a new run at any time.
        </p>
        """
        
        return await self._send_email(recipient_email, subject, html_content)
    
    async def _send_email(
        self,
        recipient_email: str,
        subject: str,
        html_content: str,
    ) -> bool:
        """Send an email via SendGrid."""
        try:
            message = Mail(
                from_email=Email(self.from_email, "HAYSTACK"),
                to_emails=To(recipient_email),
                subject=subject,
                html_content=Content("text/html", html_content),
            )
            
            response = self.client.send(message)
            
            if response.status_code in (200, 201, 202):
                logger.info(f"Email sent to {recipient_email}: {subject}")
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


# Global service instance
email_service = EmailService()
```

---

## Related Specs

- `specification/agents.md`
- `specification/tools.md`
- `specification/configuration.md`
- `specification/deployment.md`
