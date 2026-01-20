# Sprint 06: Cloud Run API

**Duration**: 1 week  
**Dependencies**: Sprint 04 (Agent Framework)  
**Goal**: Implement the FastAPI backend for job submission and status polling.

---

## Overview

> **Spec Reference**: `./specification/backend-api.md`

This sprint implements the Cloud Run API layer:
- FastAPI application with run management endpoints
- CPU Batch client for orchestrator job submission
- IAP authentication integration
- Metadata endpoints

---

## Phase 1: FastAPI Application

### Task 1.1: Create Main Application

> **Spec Reference**: `./specification/backend-api.md` (Section 9.2)

- [ ] **1.1.1** Create `api/main.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    await gcs_service.initialize()
    yield
    await database.close()

def create_app() -> FastAPI:
    app = FastAPI(
        title="HAYSTACK API",
        version="1.0.0",
        description="Iterative Knowledge-Guided Cell Prompting System",
        lifespan=lifespan,
    )
    
    # Include routers
    app.include_router(runs_router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(metadata_router, prefix="/api/v1/metadata", tags=["metadata"])
    app.include_router(health_router, prefix="/api/v1/health", tags=["health"])
    
    # Mount static frontend (only if directory exists)
    if Path("/app/frontend/out").exists():
        app.mount("/", StaticFiles(directory="/app/frontend/out", html=True))
    
    return app

app = create_app()
```

---

### Task 1.2: Implement Health Endpoints

- [ ] **1.2.1** Create `api/routes/health.py`:

```python
@router.get("/")
async def health_check():
    return {"status": "healthy"}

@router.get("/ready")
async def readiness_check():
    # Check database connectivity
    try:
        await database.execute_query("SELECT 1", ())
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(503, f"Database not ready: {e}")
```

---

## Phase 2: Run Management Endpoints

> **Spec Reference**: `./specification/backend-api.md` (Section 9.3)

### Task 2.1: Implement POST /api/v1/runs/

- [ ] **2.1.1** Create `api/routes/runs.py`:

```python
@router.post("/", response_model=RunStatusResponse)
async def create_run(
    request: CreateRunRequest,
    user_email: str = Depends(get_current_user),
):
    """Create a new HAYSTACK run."""
    run_id = f"run_{uuid4().hex[:12]}"
    
    # Create run record
    await database.create_run(
        run_id=run_id,
        user_email=user_email,
        query=request.query,
        control_strategy=request.control_strategy,
        config={
            "max_iterations": request.max_iterations or 5,
            "score_threshold": request.score_threshold or 7,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
        },
    )
    
    # Submit CPU Batch job
    try:
        job_name = await batch_client.submit_orchestrator_job(
            run_id=run_id,
            query=request.query,
            user_email=user_email,
            config=request.dict(),
        )
        await database.update_run(run_id, batch_job_name=job_name)
    except Exception as e:
        await database.update_run(run_id, status="failed", error_message=str(e))
        raise HTTPException(500, f"Failed to submit run: {e}")
    
    return RunStatusResponse(run_id=run_id, status="pending", ...)
```

- [ ] **2.1.2** Implement request validation:
  - Query minimum length: 10 characters
  - Valid control strategy
  - Optional config validation

---

### Task 2.2: Implement GET /api/v1/runs/{id}

- [ ] **2.2.1** Read run status from database:

```python
@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get status of a run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    return RunStatusResponse(
        run_id=run["run_id"],
        status=run["status"],
        current_iteration=run["current_iteration"],
        max_iterations=run["config"].get("max_iterations", 5),
        current_phase=run.get("current_phase"),
        grounding_scores=[it["grounding_score"]["composite_score"] 
                         for it in run.get("iterations", [])],
        control_strategy=run.get("control_strategy"),
        control_strategy_effective=run.get("control_strategy_effective"),
        error_message=run.get("error_message"),
        ...
    )
```

---

### Task 2.3: Implement GET /api/v1/runs/{id}/result

- [ ] **2.3.1** Generate signed GCS URLs:

```python
@router.get("/{run_id}/result", response_model=RunResultResponse)
async def get_run_result(run_id: str):
    """Get results of a completed run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    if run["status"] != "completed":
        raise HTTPException(400, "Run not completed")
    
    # Generate signed URLs (1 hour expiration)
    anndata_url = await gcs_service.generate_signed_url(
        run["output_anndata_path"], expiration=3600
    )
    report_url = await gcs_service.generate_signed_url(
        run["output_report_path"], expiration=3600
    )
    log_url = await gcs_service.generate_signed_url(
        run["output_log_path"], expiration=3600
    )
    
    return RunResultResponse(
        run_id=run_id,
        success=True,
        grounding_score=run["final_score"],
        anndata_url=anndata_url,
        report_url=report_url,
        log_url=log_url,
        ...
    )
```

---

### Task 2.4: Implement GET /api/v1/runs/

- [ ] **2.4.1** List user's runs with pagination:

```python
@router.get("/", response_model=RunListResponse)
async def list_runs(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    user_email: str = Depends(get_current_user),
):
    """List runs for the current user."""
    runs, total = await database.list_runs(
        user_email=user_email,
        page=page,
        page_size=page_size,
        status_filter=status,
    )
    return RunListResponse(runs=runs, total=total, page=page, page_size=page_size)
```

---

### Task 2.5: Implement POST /api/v1/runs/{id}/cancel

- [ ] **2.5.1** Cancel running job:

```python
@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel a running run."""
    run = await database.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    
    if run["status"] not in ["pending", "running"]:
        raise HTTPException(400, "Run cannot be cancelled")
    
    # Cancel Batch job
    if run.get("batch_job_name"):
        await batch_client.cancel_job(run["batch_job_name"])
    
    await database.update_run(run_id, status="cancelled")
    return {"status": "cancelled"}
```

---

### Task 2.6: Implement POST /api/v1/runs/validate-control-strategy

- [ ] **2.6.1** Check synthetic control availability:

```python
@router.post("/validate-control-strategy", response_model=ControlStrategyValidation)
async def validate_control_strategy(request: ControlStrategyValidationRequest):
    """Validate whether synthetic control strategy is possible."""
    # Quick resolution of query to check control availability
    # Returns recommendation based on matched control cells
    ...
```

---

## Phase 3: CPU Batch Client

### Task 3.1: Implement Orchestrator Batch Client

> **Spec Reference**: `./specification/backend-api.md` (Section 9.4)

- [ ] **3.1.1** Create `api/services/batch.py`:

```python
class OrchestratorBatchClient:
    """Client for submitting CPU orchestrator Batch jobs."""
    
    async def submit_orchestrator_job(
        self,
        run_id: str,
        query: str,
        user_email: str,
        config: dict,
    ) -> str:
        """Submit orchestrator job and return job name."""
        job_id = f"haystack-{run_id}"
        
        # Configure container
        container = types.Runnable.Container(
            image_uri=f"{self.registry}/haystack-orchestrator:latest",
            commands=["python", "-m", "orchestrator.main"],
        )
        
        # Configure environment
        task = types.TaskSpec(
            runnables=[types.Runnable(container=container)],
            environment=types.Environment(
                variables={
                    "RUN_ID": run_id,
                    "USER_EMAIL": user_email,
                    "CONTROL_STRATEGY": config.get("control_strategy", "synthetic_control"),
                    "MAX_ITERATIONS": str(config.get("max_iterations", 5)),
                }
            ),
            max_run_duration="3600s",  # 1 hour timeout
        )
        
        # Configure machine
        allocation_policy = types.AllocationPolicy(
            instances=[
                types.AllocationPolicy.InstancePolicyOrTemplate(
                    policy=types.AllocationPolicy.InstancePolicy(
                        machine_type="e2-standard-4",
                    )
                )
            ],
        )
        
        # Submit job
        ...
```

---

## Phase 4: IAP Integration

### Task 4.1: Implement Authentication

- [ ] **4.1.1** Create `api/auth.py`:

```python
async def get_current_user(request: Request) -> str:
    """Extract user email from IAP header."""
    # Production: Get from IAP header
    user_email = request.headers.get("X-Goog-Authenticated-User-Email")
    if user_email:
        # Format: accounts.google.com:user@example.com
        return user_email.split(":")[-1]
    
    # Development fallback
    if settings.environment == "development":
        return "dev@example.com"
    
    raise HTTPException(401, "Authentication required")
```

---

## Phase 5: Metadata Endpoints

### Task 5.1: Implement Metadata Routes

- [ ] **5.1.1** Create `api/routes/metadata.py`:

```python
@router.get("/perturbations")
async def list_perturbations(dataset: Optional[str] = None):
    """List available perturbations."""
    return await database.list_perturbations(dataset)

@router.get("/cell-types")
async def list_cell_types(dataset: Optional[str] = None):
    """List available cell types."""
    return await database.list_cell_types(dataset)

@router.get("/datasets")
async def list_datasets():
    """List available datasets."""
    return ["parse_pbmc", "openproblems", "tabula_sapiens"]
```

---

## Phase 6: Testing

### Task 6.1: API Tests

- [ ] **6.1.1** Test create run endpoint
- [ ] **6.1.2** Test get run status endpoint
- [ ] **6.1.3** Test cancel run endpoint
- [ ] **6.1.4** Test authentication handling

---

## Definition of Done

- [ ] All API endpoints implemented
- [ ] CPU Batch client submits orchestrator jobs
- [ ] IAP authentication works
- [ ] Signed URL generation works
- [ ] API tests pass

---

## Next Sprint

**Sprint 07: Frontend** - Implement the Next.js frontend for user interaction.
