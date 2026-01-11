# Backend API Specification

### 9.1 Project Structure

The backend is split into two components:
1. **`api/`** — Thin FastAPI layer running on Cloud Run (job submission, status queries)
2. **`orchestrator/`** — Agent workflow running in CPU Batch job

```
haystack/
├── api/                        # Cloud Run API (thin layer)
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory
│   ├── config.py               # Dynaconf settings
│   ├── dependencies.py         # IAP user extraction
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── runs.py             # /api/v1/runs endpoints
│   │   ├── metadata.py         # /api/v1/metadata endpoints
│   │   └── health.py           # Health check
│   ├── models/
│   │   ├── __init__.py
│   │   └── runs.py             # Run-related Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py         # Cloud SQL client (read status)
│   │   ├── gcs.py              # GCS signed URLs
│   │   └── batch.py            # GCP Batch job submission
│   └── tests/
│       └── ...
│
├── orchestrator/               # CPU Batch Job (agent workflow)
│   ├── __init__.py
│   ├── main.py                 # Entrypoint for Batch job
│   ├── config.py               # Dynaconf settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Main orchestrator loop
│   │   ├── query_understanding.py
│   │   ├── prompt_generation.py
│   │   └── grounding_evaluation.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── database_tools.py   # Cell retrieval tools
│   │   ├── knowledge_tools.py  # External API tools
│   │   ├── inference_tools.py  # GPU job submission
│   │   └── enrichment_tools.py # Pathway enrichment
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py         # Cloud SQL client (update status)
│   │   ├── gcs.py              # GCS read/write
│   │   ├── batch.py            # GPU Batch job submission
│   │   ├── email.py            # SendGrid notifications
│   │   └── biological_apis.py  # KEGG, Reactome, etc.
│   └── tests/
│       └── ...
│
├── inference/                  # GPU Batch Job (STACK only)
│   ├── __init__.py
│   ├── run_inference.py        # Entrypoint for GPU job
│   └── stack_wrapper.py        # STACK model loading/inference
│
├── shared/                     # Shared code between components
│   ├── __init__.py
│   ├── models/                 # Shared Pydantic models
│   │   ├── runs.py
│   │   ├── cells.py
│   │   └── queries.py
│   └── config.py               # Shared configuration
│
├── docker/
│   ├── Dockerfile.api          # Cloud Run container
│   ├── Dockerfile.orchestrator # CPU Batch container
│   └── Dockerfile.inference    # GPU Batch container
│
└── pyproject.toml
```

### 9.2 Main Application

```python
# backend/main.py
"""FastAPI application factory for HAYSTACK."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.routes import runs, metadata, health
from api.services.database import database
from api.services.gcs import gcs_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await database.connect()
    await gcs_service.initialize()
    yield
    # Shutdown
    await database.close()


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    This is a thin API layer that:
    - Submits CPU Batch jobs for orchestrator workflow
    - Reads run status from Cloud SQL
    - Cancels Batch jobs via API
    - Serves static frontend
    
    The actual agent workflow runs in a separate CPU Batch job.
    """
    app = FastAPI(
        title="HAYSTACK API",
        description="Iterative Knowledge-Guided Cell Prompting System",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API routes
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(metadata.router, prefix="/api/v1/metadata", tags=["metadata"])
    
    # Serve static frontend (production)
    frontend_dir = os.environ.get("FRONTEND_OUT_DIR", "/app/frontend/out")
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    
    return app


app = create_app()
```

### 9.3 API Endpoints (Cloud Run)

The Cloud Run API is intentionally thin — it submits Batch jobs and queries status from Cloud SQL.
It does not expose endpoints for browsing cell sets; those are computed dynamically inside the orchestrator.

```python
# api/routes/runs.py
"""Run management endpoints (thin layer)."""

from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from shared.models.runs import (
    CreateRunRequest,
    RunStatusResponse,
    RunResultResponse,
    RunListResponse,
)
from api.services.database import database
from api.services.batch import batch_client
from api.services.gcs import gcs_service
from api.dependencies import get_current_user

router = APIRouter()


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid4().hex[:12]}"


@router.post("/", response_model=RunStatusResponse)
async def create_run(
    request: CreateRunRequest,
    user_email: str = Depends(get_current_user),
):
    """
    Create a new HAYSTACK run.
    
    Submits a CPU Batch job to run the orchestrator workflow.
    Returns immediately with run_id for status polling.
    """
    run_id = generate_run_id()
    
    # Create run record in database
    await database.create_run(
        run_id=run_id,
        user_email=user_email,
        query=request.query,
        config=request.model_dump(),
        status="pending",
    )
    
    # Submit CPU Batch job for orchestrator
    try:
        job_name = await batch_client.submit_orchestrator_job(
            run_id=run_id,
            query=request.query,
            user_email=user_email,
            config=request.model_dump(),
        )
        
        # Store Batch job name for cancellation
        await database.update_run(
            run_id=run_id,
            batch_job_name=job_name,
        )
        
    except Exception as e:
        await database.update_run(
            run_id=run_id,
            status="failed",
            error_message=f"Failed to submit Batch job: {e}",
        )
        raise HTTPException(500, f"Failed to submit run: {e}")
    
    return RunStatusResponse(
        run_id=run_id,
        status="pending",
        current_iteration=0,
        max_iterations=request.max_iterations or 5,
        current_phase=None,
        grounding_scores=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        user_email=user_email,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get current status of a run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    return RunStatusResponse(
        run_id=run_id,
        status=run["status"],
        current_iteration=len(run.get("iterations", [])),
        max_iterations=run["config"].get("max_iterations", 5),
        current_score=run.get("final_score"),
        message=run.get("termination_reason", ""),
    )


@router.get("/{run_id}/result", response_model=RunResultResponse)
async def get_run_result(run_id: str):
    """Get results of a completed run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    if run["status"] != "completed":
        raise HTTPException(400, f"Run {run_id} is not completed (status: {run['status']})")
    
    # Generate signed URLs for downloads
    anndata_url = await gcs_service.get_signed_url(run["output_anndata_path"])
    report_url = await gcs_service.get_signed_url(run["output_report_path"])
    log_url = await gcs_service.get_signed_url(run["output_log_path"])
    
    return RunResultResponse(
        run_id=run_id,
        success=True,
        grounding_score=run["final_score"],
        termination_reason=run["termination_reason"],
        # ... other fields from final iteration
        anndata_url=anndata_url,
        report_url=report_url,
        log_url=log_url,
    )


@router.get("/", response_model=RunListResponse)
async def list_runs(
    user_email: str = Depends(get_current_user),
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
):
    """List runs for current user."""
    runs, total = await database.list_runs(
        user_email=user_email,
        page=page,
        page_size=page_size,
        status_filter=status,
    )
    
    return RunListResponse(
        runs=runs,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel a running run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    if run["status"] not in ("pending", "running"):
        raise HTTPException(400, f"Run {run_id} cannot be cancelled (status: {run['status']})")
    
    # Set cancellation flag in database
    await database.update_run(run_id=run_id, status="cancelled")
    
    # Cancel the Batch job if it exists
    batch_job_name = run.get("batch_job_name")
    if batch_job_name:
        try:
            await batch_client.cancel_job(batch_job_name)
        except Exception as e:
            # Job may have already completed or failed
            logger.warning(f"Failed to cancel Batch job {batch_job_name}: {e}")
    
    return {"message": f"Run {run_id} cancelled"}
```

### 9.4 Batch Client (Cloud Run)

The Cloud Run API uses this client to submit and cancel CPU orchestrator jobs.

```python
# api/services/batch.py
"""GCP Batch client for submitting orchestrator jobs."""

import logging
from google.cloud import batch_v1

from api.config import settings

logger = logging.getLogger(__name__)


class BatchClient:
    """Client for submitting CPU orchestrator Batch jobs."""
    
    def __init__(self):
        self.client = batch_v1.BatchServiceAsyncClient()
        self.project = settings.gcp_project_id
        self.region = settings.batch.region
    
    async def submit_orchestrator_job(
        self,
        run_id: str,
        query: str,
        user_email: str,
        config: dict,
    ) -> str:
        """
        Submit a CPU Batch job to run the orchestrator agent.
        
        Args:
            run_id: Unique run identifier
            query: User's natural language query
            user_email: User's email for notifications
            config: Run configuration
        
        Returns:
            Batch job name for tracking/cancellation
        """
        job_name = f"haystack-orchestrator-{run_id}"
        
        # Container configuration
        container = batch_v1.Runnable.Container(
            image_uri=settings.batch.orchestrator_image,
            commands=["python", "-m", "orchestrator.main"],
            options=f"--env RUN_ID={run_id} --env USER_EMAIL={user_email}",
        )
        
        runnable = batch_v1.Runnable(container=container)
        
        # Task specification
        task_spec = batch_v1.TaskSpec(
            runnables=[runnable],
            max_retry_count=0,  # No retries - let agent handle errors
            max_run_duration="7200s",  # 2 hour max
        )
        
        # Resource requirements (CPU only)
        resources = batch_v1.ComputeResource(
            cpu_milli=4000,   # 4 vCPUs
            memory_mib=16384,  # 16 GB RAM
        )
        task_spec.compute_resource = resources
        
        # Task group
        task_group = batch_v1.TaskGroup(
            task_count=1,
            task_spec=task_spec,
        )
        
        # Allocation policy (CPU machine)
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy(
            machine_type=settings.batch.orchestrator_machine_type,
        )
        
        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
            policy=instance_policy,
        )
        
        allocation_policy = batch_v1.AllocationPolicy(
            instances=[instances],
            location=batch_v1.AllocationPolicy.LocationPolicy(
                allowed_locations=[f"regions/{self.region}"],
            ),
        )
        
        # Network configuration for Cloud SQL access
        network_policy = batch_v1.AllocationPolicy.NetworkPolicy(
            network_interfaces=[
                batch_v1.AllocationPolicy.NetworkInterface(
                    network=f"projects/{self.project}/global/networks/default",
                    subnetwork=f"projects/{self.project}/regions/{self.region}/subnetworks/default",
                )
            ]
        )
        allocation_policy.network = network_policy
        
        # Service account
        service_account = batch_v1.ServiceAccount(
            email=settings.batch.orchestrator_service_account,
        )
        allocation_policy.service_account = service_account
        
        # Create job
        job = batch_v1.Job(
            task_groups=[task_group],
            allocation_policy=allocation_policy,
            logs_policy=batch_v1.LogsPolicy(
                destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
            ),
            labels={
                "haystack-run-id": run_id,
                "haystack-component": "orchestrator",
            },
        )
        
        # Submit job
        request = batch_v1.CreateJobRequest(
            parent=f"projects/{self.project}/locations/{self.region}",
            job_id=job_name,
            job=job,
        )
        
        response = await self.client.create_job(request=request)
        logger.info(f"Submitted orchestrator job: {response.name}")
        
        return response.name
    
    async def cancel_job(self, job_name: str) -> None:
        """Cancel a Batch job."""
        request = batch_v1.DeleteJobRequest(name=job_name)
        await self.client.delete_job(request=request)
        logger.info(f"Cancelled job: {job_name}")


# Global client instance
batch_client = BatchClient()
```

### 9.5 IAP User Extraction

```python
# api/dependencies.py
"""FastAPI dependencies including IAP user extraction."""

from fastapi import Request, HTTPException
from api.config import settings


def get_current_user(request: Request) -> str:
    """
    Extract user email from GCP IAP headers.
    
    IAP sets the following headers:
    - X-Goog-Authenticated-User-Email: accounts.google.com:user@example.com
    - X-Goog-Authenticated-User-ID: accounts.google.com:123456789
    
    Returns:
        User email address
    
    Raises:
        HTTPException: If IAP headers are missing
    """
    iap_email = request.headers.get("X-Goog-Authenticated-User-Email")
    
    if not iap_email:
        # In development, fall back to a default or query param
        if settings.debug:
            return request.query_params.get("user_email", "dev@arc.institute")
        raise HTTPException(401, "IAP authentication required")
    
    # Remove the "accounts.google.com:" prefix
    if ":" in iap_email:
        return iap_email.split(":", 1)[1]
    return iap_email
```

---

## Related Specs

- `specification/data-models.md`
- `specification/frontend.md`
- `specification/orchestrator.md`
