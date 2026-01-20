# Sprint 08: Orchestrator Batch Job

**Duration**: 1-2 weeks  
**Dependencies**: Sprint 04 (Agent Framework), Sprint 05 (STACK Inference), Sprint 06 (Cloud Run API)  
**Goal**: Finalize the orchestrator CPU batch job entrypoint and email notifications.

---

## Overview

> **Spec Reference**: `./specification/orchestrator.md`

This sprint implements:
- Orchestrator batch job entrypoint
- Email notification service (SendGrid)
- Output generation (AnnData, reports, logs)
- Error handling and recovery

---

## Phase 1: Orchestrator Entrypoint

### Task 1.1: Implement Main Entrypoint

> **Spec Reference**: `./specification/orchestrator.md` (Section 10.1)

- [ ] **1.1.1** Create `orchestrator/main.py`:

```python
"""
HAYSTACK Orchestrator - CPU Batch Job Entrypoint.

Environment Variables:
    RUN_ID: Unique run identifier
    USER_EMAIL: User email for notifications
    CONTROL_STRATEGY: synthetic_control or query_as_control
"""

import asyncio
import logging
import os
import sys

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
    run_id = os.environ.get("RUN_ID")
    user_email = os.environ.get("USER_EMAIL")
    control_strategy = os.environ.get("CONTROL_STRATEGY", "synthetic_control")
    
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
            raise ValueError(f"Run {run_id} not found in database")
        
        query = run["raw_query"]
        config = run["config"]
        
        # Update status to running
        await database.update_run(run_id, status="running", current_phase="query_analysis")
        
        # Create and run orchestrator agent
        agent = OrchestratorAgent(
            run_id=run_id,
            query=query,
            user_email=user_email,
            config=config,
            control_strategy=control_strategy,
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
            logger.info(f"Run {run_id} completed with score {result.final_score}")
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
        
        await database.update_run(
            run_id=run_id,
            status="failed",
            error_message=str(e),
        )
        
        # Try to send failure email
        try:
            await email_service.send_run_failed(
                recipient_email=user_email,
                run_id=run_id,
                query=run.get("raw_query", ""),
                error_message=str(e),
            )
        except Exception:
            pass
        
        sys.exit(1)
        
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Task 1.2: Implement Cancellation Check

- [ ] **1.2.1** Add cancellation polling in iteration loop:

```python
async def _check_cancellation(self) -> bool:
    """Check if run has been cancelled by user."""
    run = await database.get_run(self.run_id)
    return run["status"] == "cancelled"
```

- [ ] **1.2.2** Check before each phase transition
- [ ] **1.2.3** Clean up resources on cancellation

---

## Phase 2: Email Notification Service

### Task 2.1: Implement SendGrid Service

- [ ] **2.1.1** Create `orchestrator/services/email.py`:

```python
"""
Email notification service using SendGrid.

Sends notifications for:
- Run completed successfully
- Run failed with error
- Run cancelled by user
"""

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from dynaconf import settings


class EmailService:
    """SendGrid email client for HAYSTACK notifications."""
    
    def __init__(self):
        self.client = None
        self.from_email = None
    
    def initialize(self):
        """Initialize SendGrid client."""
        api_key = settings.sendgrid.api_key
        self.from_email = settings.sendgrid.from_email
        self.client = SendGridAPIClient(api_key)
    
    async def send_run_completed(
        self,
        recipient_email: str,
        run_id: str,
        query: str,
        grounding_score: int,
    ):
        """Send run completion notification."""
        subject = f"HAYSTACK Run Complete - Score {grounding_score}/10"
        
        html_content = f"""
        <h2>Your HAYSTACK run has completed!</h2>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Grounding Score:</strong> {grounding_score}/10</p>
        <p>View your results: <a href="{settings.app_url}/runs/{run_id}">
            {settings.app_url}/runs/{run_id}
        </a></p>
        """
        
        await self._send_email(recipient_email, subject, html_content)
    
    async def send_run_failed(
        self,
        recipient_email: str,
        run_id: str,
        query: str,
        error_message: str,
    ):
        """Send run failure notification."""
        subject = f"HAYSTACK Run Failed - {run_id}"
        
        html_content = f"""
        <h2>Your HAYSTACK run has failed</h2>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Error:</strong> {error_message}</p>
        <p>Please contact support if this issue persists.</p>
        """
        
        await self._send_email(recipient_email, subject, html_content)
    
    async def _send_email(self, to: str, subject: str, html_content: str):
        """Send email via SendGrid."""
        message = Mail(
            from_email=self.from_email,
            to_emails=to,
            subject=subject,
            html_content=html_content,
        )
        response = self.client.send(message)
        return response


email_service = EmailService()
```

---

### Task 2.2: Create Email Templates

- [ ] **2.2.1** Design HTML email templates:
  - `run_completed.html` - Success with score and download links
  - `run_failed.html` - Failure with error details
  - `run_cancelled.html` - Cancellation confirmation

- [ ] **2.2.2** Add HAYSTACK branding to templates
- [ ] **2.2.3** Test email rendering in common clients

---

## Phase 3: Output Generation

### Task 3.1: Generate Final AnnData

- [ ] **3.1.1** Create `orchestrator/output/anndata_generator.py`:

```python
async def generate_final_anndata(
    run_id: str,
    predictions: AnnData,
    query: StructuredQuery,
    iterations: list[IterationRecord],
) -> str:
    """Generate final annotated AnnData with all metadata."""
    # Add run metadata to .uns
    predictions.uns["haystack"] = {
        "run_id": run_id,
        "query": query.raw_query,
        "task_type": query.task_type.value,
        "iterations": len(iterations),
        "final_score": iterations[-1].grounding_score.composite_score,
        "control_strategy": iterations[-1].control_strategy.value,
    }
    
    # Add iteration history
    predictions.uns["haystack"]["iterations"] = [
        {
            "number": it.iteration_number,
            "score": it.grounding_score.composite_score,
            "strategy": it.selected_prompt.strategy,
        }
        for it in iterations
    ]
    
    # Write to GCS
    gcs_path = f"gs://haystack-results/runs/{run_id}/output.h5ad"
    await gcs_service.upload_anndata(predictions, gcs_path)
    
    return gcs_path
```

---

### Task 3.2: Generate Interpretation Report

- [ ] **3.2.1** Create `orchestrator/output/report_generator.py`:

```python
async def generate_report(
    run_id: str,
    query: StructuredQuery,
    final_score: Union[GroundingScore, ObservationalGroundingScore],
    iterations: list[IterationRecord],
) -> str:
    """Generate markdown interpretation report."""
    report = f"""# HAYSTACK Run Report

## Query
{query.raw_query}

## Results Summary
- **Task Type**: {query.task_type.value}
- **Final Score**: {final_score.composite_score}/10
- **Iterations**: {len(iterations)}
- **Control Strategy**: {iterations[-1].control_strategy.value}

## Score Breakdown
"""
    
    if isinstance(final_score, GroundingScore):
        report += f"""
- Pathway Coherence: {final_score.pathway_coherence}/10
- Target Activation: {final_score.target_activation}/10
- Literature Support: {final_score.literature_support}/10
- Network Coherence: {final_score.network_coherence}/10

## Enriched Pathways
{format_pathways(final_score.enriched_pathways)}

## Top Differentially Expressed Genes
**Upregulated**: {', '.join(final_score.de_genes_up[:20])}
**Downregulated**: {', '.join(final_score.de_genes_down[:20])}
"""
    else:
        report += f"""
- Marker Gene Expression: {final_score.marker_gene_expression}/10
- Tissue Signature Match: {final_score.tissue_signature_match}/10
- Donor Effect Capture: {final_score.donor_effect_capture}/10
- Cell Type Coherence: {final_score.cell_type_coherence}/10
"""
    
    # Write to GCS
    gcs_path = f"gs://haystack-results/runs/{run_id}/report.md"
    await gcs_service.write_markdown(report, gcs_path)
    
    return gcs_path
```

---

### Task 3.3: Generate Structured Log

- [ ] **3.3.1** Create `orchestrator/output/log_generator.py`:

```python
async def generate_log(
    run_id: str,
    run: HaystackRun,
) -> str:
    """Generate structured JSON log for debugging."""
    log = {
        "run_id": run.run_id,
        "user_email": run.user_email,
        "start_time": run.start_time.isoformat(),
        "end_time": run.end_time.isoformat() if run.end_time else None,
        "status": run.status,
        "control_strategy": run.control_strategy.value,
        "control_strategy_effective": run.control_strategy_effective.value if run.control_strategy_effective else None,
        "config": run.config,
        "query": {
            "raw": run.raw_query,
            "structured": run.structured_query.dict() if run.structured_query else None,
        },
        "iterations": [
            {
                "number": it.iteration_number,
                "duration_seconds": it.duration_seconds,
                "prompt_strategy": it.selected_prompt.strategy,
                "grounding_score": it.grounding_score.composite_score,
                "control_strategy": it.control_strategy.value,
            }
            for it in run.iterations
        ],
        "final_score": run.final_score,
        "termination_reason": run.termination_reason,
    }
    
    gcs_path = f"gs://haystack-results/runs/{run_id}/log.json"
    await gcs_service.write_json(log, gcs_path)
    
    return gcs_path
```

---

## Phase 4: Error Handling and Recovery

### Task 4.1: Implement Error Handling

- [ ] **4.1.1** Create error categories:
  - `QueryParsingError` - Failed to parse user query
  - `EntityResolutionError` - Failed to resolve cell type or perturbation
  - `RetrievalError` - No suitable prompt cells found
  - `InferenceError` - STACK inference failed
  - `EvaluationError` - Grounding evaluation failed

- [ ] **4.1.2** Log errors with structured context
- [ ] **4.1.3** Update database with error details

---

### Task 4.2: Implement Graceful Degradation

- [ ] **4.2.1** Handle missing control cells:
  - Fallback from synthetic_control to query_as_control
  - Log warning but continue execution

- [ ] **4.2.2** Handle low grounding scores:
  - Continue iterations up to max
  - Return partial results even if below threshold

---

## Definition of Done

- [ ] Orchestrator entrypoint handles all environment variables
- [ ] Email notifications sent for all run states
- [ ] Output generation creates valid AnnData, report, and log
- [ ] Error handling categorizes and logs all errors
- [ ] Cancellation handled gracefully

---

## Next Sprint

**Sprint 09: Docker Containers** - Build and configure all Docker containers.
