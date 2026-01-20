# Sprint 05: STACK Inference Integration

**Duration**: 1 week  
**Dependencies**: Sprint 04 (Agent Framework)  
**Goal**: Implement the GPU Batch job for STACK model inference.

---

## Overview

> **Spec Reference**: `./specification/tools.md` (Section 6.4), `./specification/deployment.md`

This sprint implements STACK inference:
- GPU Batch job container with STACK model
- Batch client for job submission and monitoring
- Inference tool for the orchestrator
- Control strategy handling (single vs. paired inference)

---

## Phase 1: Inference Container

### Task 1.1: Create Dockerfile.inference

> **Spec Reference**: `./specification/deployment.md` (Section 12.3)

- [ ] **1.1.1** Create `docker/Dockerfile.inference`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# Install STACK
RUN git clone https://github.com/arcinstitute/STACK.git /app/stack
RUN pip install -e /app/stack

# Install dependencies
RUN pip install \
    google-cloud-storage \
    scanpy \
    anndata \
    scipy \
    tqdm

# Copy inference script
COPY inference /app/inference

WORKDIR /app
ENTRYPOINT ["python", "-m", "inference.run_inference"]
```

- [ ] **1.1.2** Test container builds successfully
- [ ] **1.1.3** Verify STACK model loads correctly

---

### Task 1.2: Implement run_inference.py

> **Spec Reference**: `./specification/deployment.md` (Section 12.4)

- [ ] **1.2.1** Create `inference/run_inference.py`:

```python
"""
STACK inference script for GCP Batch.

Usage:
    python run_inference.py \
        --prompt-gcs gs://bucket/batch-io/run_id/prompt.h5ad \
        --query-gcs gs://bucket/batch-io/run_id/query.h5ad \
        --output-gcs gs://bucket/batch-io/run_id/predictions.h5ad \
        --model-gcs gs://bucket/models/stack_v1/model.pt \
        --genelist-gcs gs://bucket/models/stack_v1/genelist.txt \
        --diffusion-steps 5 \
        --batch-size 32
"""

import argparse
import tempfile
from pathlib import Path

def download_from_gcs(gcs_path: str, local_path: Path):
    """Download file from GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_name = "/".join(gcs_path.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))

def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload file to GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_name = "/".join(gcs_path.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-gcs", required=True)
    parser.add_argument("--query-gcs", required=True)
    parser.add_argument("--output-gcs", required=True)
    parser.add_argument("--model-gcs", required=True)
    parser.add_argument("--genelist-gcs", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Download inputs
        prompt_path = tmpdir / "prompt.h5ad"
        query_path = tmpdir / "query.h5ad"
        model_path = tmpdir / "model.pt"
        genelist_path = tmpdir / "genelist.txt"
        output_path = tmpdir / "predictions.h5ad"
        
        download_from_gcs(args.prompt_gcs, prompt_path)
        download_from_gcs(args.query_gcs, query_path)
        download_from_gcs(args.model_gcs, model_path)
        download_from_gcs(args.genelist_gcs, genelist_path)
        
        # Load data
        import scanpy as sc
        prompt_adata = sc.read_h5ad(prompt_path)
        query_adata = sc.read_h5ad(query_path)
        
        # Load STACK model
        from stack import STACKPredictor
        predictor = STACKPredictor.from_checkpoint(
            model_path, genelist_path
        )
        
        # Run inference
        predictions = predictor.predict(
            prompt_adata=prompt_adata,
            query_adata=query_adata,
            T=args.diffusion_steps,
            batch_size=args.batch_size,
        )
        
        # Save and upload
        predictions.write_h5ad(output_path)
        upload_to_gcs(output_path, args.output_gcs)

if __name__ == "__main__":
    main()
```

---

## Phase 2: GPU Batch Client

### Task 2.1: Implement BatchClient

> **Spec Reference**: `./specification/deployment.md` (Section 12.4)

- [ ] **2.1.1** Create `orchestrator/services/batch.py`:

```python
@dataclass
class BatchJobConfig:
    run_id: str
    prompt_gcs_path: str
    query_gcs_path: str
    output_gcs_path: str
    model_gcs_path: str = "gs://haystack-models/stack_v1/model.pt"
    genelist_gcs_path: str = "gs://haystack-models/stack_v1/genelist.txt"
    diffusion_steps: int = 5
    batch_size: int = 32

class GPUBatchClient:
    """Client for submitting STACK inference jobs to GCP Batch."""
    
    def __init__(self, config: BatchConfig):
        self.client = batch_v1.BatchServiceAsyncClient()
        self.project = config.project_id
        self.region = config.region
        self.service_account = config.inference_service_account
    
    async def submit_inference_job(self, job_config: BatchJobConfig) -> str:
        """Submit inference job and return job name."""
    
    async def get_job_status(self, job_name: str) -> tuple[str, str | None]:
        """Get job status and optional error message."""
    
    async def wait_for_completion(
        self, 
        job_name: str, 
        poll_interval: int = 10,
        timeout: int = 1800,
    ) -> tuple[bool, str | None]:
        """Wait for job to complete with polling."""
    
    async def cancel_job(self, job_name: str) -> None:
        """Cancel a running job."""
```

---

### Task 2.2: Implement Job Submission

- [ ] **2.2.1** Configure GPU allocation (A100 80GB):

```python
allocation_policy = types.AllocationPolicy(
    instances=[
        types.AllocationPolicy.InstancePolicyOrTemplate(
            policy=types.AllocationPolicy.InstancePolicy(
                machine_type="a2-highgpu-1g",
                accelerators=[
                    types.AllocationPolicy.Accelerator(
                        type_="nvidia-tesla-a100",
                        count=1,
                    )
                ],
            )
        )
    ],
    location=types.AllocationPolicy.LocationPolicy(
        allowed_locations=[f"regions/{self.region}"],
    ),
)
```

- [ ] **2.2.2** Configure container and arguments
- [ ] **2.2.3** Set job timeout (30 minutes default)

---

### Task 2.3: Implement Job Monitoring

- [ ] **2.3.1** Implement polling loop with configurable interval
- [ ] **2.3.2** Handle job states: QUEUED, RUNNING, SUCCEEDED, FAILED
- [ ] **2.3.3** Extract error messages from failed jobs
- [ ] **2.3.4** Implement cancellation

---

## Phase 3: Inference Tool

### Task 3.1: Implement run_stack_inference Tool

- [ ] **3.1.1** Create `orchestrator/tools/inference_tools.py`:

```python
@tool
async def run_stack_inference(
    run_id: str,
    iteration: int,
    prompt_cell_indices: str,  # JSON array
    query_cell_indices: str,   # JSON array
    control_strategy: str,
    paired_control_indices: Optional[str] = None,
) -> str:
    """
    Run STACK inference via GPU Batch job.
    
    For synthetic_control strategy, runs two inference jobs:
    1. Perturbed prompt → predictions
    2. Control prompt → control predictions
    
    For query_as_control strategy, runs single inference job.
    """
```

- [ ] **3.1.2** Prepare input data:
  - Extract cells from atlas H5AD by indices
  - Write prompt.h5ad and query.h5ad to GCS

- [ ] **3.1.3** Handle synthetic control:
  - Submit two jobs (perturbed + control)
  - Wait for both to complete
  - Return paths to both prediction files

- [ ] **3.1.4** Handle query-as-control:
  - Submit single job
  - Return path to prediction file

---

### Task 3.2: Integrate with Orchestrator

- [ ] **3.2.1** Call inference from iteration loop
- [ ] **3.2.2** Update run status during inference
- [ ] **3.2.3** Handle inference failures gracefully
- [ ] **3.2.4** Store prediction paths in IterationRecord

---

## Phase 4: Testing

### Task 4.1: Unit Tests

- [ ] **4.1.1** Test GCS upload/download functions
- [ ] **4.1.2** Test batch job configuration creation
- [ ] **4.1.3** Test job monitoring with mocked client

### Task 4.2: Integration Tests

- [ ] **4.2.1** Test container with sample data
- [ ] **4.2.2** Test end-to-end inference pipeline
- [ ] **4.2.3** Test control strategy handling

---

## Definition of Done

- [ ] Inference container builds and runs correctly
- [ ] STACK model loads and produces valid predictions
- [ ] Batch client submits and monitors jobs
- [ ] Synthetic control runs two inference jobs
- [ ] Query-as-control runs single inference job
- [ ] Integration tests pass

---

## Next Sprint

**Sprint 06: Cloud Run API** - Implement the FastAPI backend for job submission and status polling.
