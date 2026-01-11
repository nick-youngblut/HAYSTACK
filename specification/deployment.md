# Deployment

HAYSTACK uses three separate containers:
1. **API Container** — Cloud Run (thin API layer + frontend)
2. **Orchestrator Container** — CPU Batch job (agent workflow)
3. **Inference Container** — GPU Batch job (STACK model)

### 12.1 API Container (Cloud Run)

```dockerfile
# docker/Dockerfile.api
# Cloud Run API + frontend serving

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN rm -f .env .env.* || true
RUN npm run build

# Stage 2: Python API
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install API dependencies
COPY api /app/api
COPY shared /app/shared
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[api]"

# Copy built frontend
COPY --from=frontend-build /app/frontend/out /app/frontend/out

# Runtime configuration
ENV FRONTEND_OUT_DIR=/app/frontend/out PORT=8080

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 12.2 Orchestrator Container (CPU Batch)

```dockerfile
# docker/Dockerfile.orchestrator
# CPU Batch job for agent workflow

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install orchestrator dependencies
COPY orchestrator /app/orchestrator
COPY shared /app/shared
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[orchestrator]"

# Entrypoint
CMD ["python", "-m", "orchestrator.main"]
```

### 12.3 Inference Container (GPU Batch)

```dockerfile
# docker/Dockerfile.inference
# GPU Batch job for STACK inference

# Base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and install STACK
RUN git clone https://github.com/arcinstitute/STACK.git /app/stack
WORKDIR /app/stack
RUN pip install --no-cache-dir -e .

# Install additional dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage>=2.14.0 \
    scanpy>=1.10.0 \
    anndata>=0.10.0

# Copy inference script
COPY inference /app/inference

WORKDIR /app
ENTRYPOINT ["python", "-m", "inference.run_inference"]
```

### 12.4 STACK Inference Script

```python
# stack_inference/run_inference.py
"""
STACK inference script for GCP Batch.

Usage:
    python run_inference.py \
        --prompt-gcs gs://bucket/batch-io/run_id/prompt.h5ad \
        --query-gcs gs://bucket/batch-io/run_id/query.h5ad \
        --output-gcs gs://bucket/batch-io/run_id/predictions.h5ad \
        --model-gcs gs://bucket/models/stack/stack_large.pt \
        --genelist-gcs gs://bucket/models/stack/genelist.json \
        --diffusion-steps 5 \
        --batch-size 32
"""

import argparse
import sys
import tempfile
from pathlib import Path

import scanpy as sc
import torch
from google.cloud import storage


def download_from_gcs(gcs_path: str, local_path: Path) -> None:
    """Download file from GCS."""
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_name = "/".join(gcs_path.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))
    print(f"Downloaded {gcs_path} to {local_path}")


def upload_to_gcs(local_path: Path, gcs_path: str) -> None:
    """Upload file to GCS."""
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_name = "/".join(gcs_path.split("/")[3:])
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded {local_path} to {gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="Run STACK inference")
    parser.add_argument("--prompt-gcs", required=True, help="GCS path to prompt cells H5AD")
    parser.add_argument("--query-gcs", required=True, help="GCS path to query cells H5AD")
    parser.add_argument("--output-gcs", required=True, help="GCS path for output predictions")
    parser.add_argument("--model-gcs", required=True, help="GCS path to STACK model checkpoint")
    parser.add_argument("--genelist-gcs", required=True, help="GCS path to gene list JSON")
    parser.add_argument("--diffusion-steps", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Create temp directory for local files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Download inputs from GCS
        prompt_path = tmpdir / "prompt.h5ad"
        query_path = tmpdir / "query.h5ad"
        model_path = tmpdir / "model.pt"
        genelist_path = tmpdir / "genelist.json"
        output_path = tmpdir / "predictions.h5ad"
        
        print("Downloading inputs from GCS...")
        download_from_gcs(args.prompt_gcs, prompt_path)
        download_from_gcs(args.query_gcs, query_path)
        download_from_gcs(args.model_gcs, model_path)
        download_from_gcs(args.genelist_gcs, genelist_path)
        
        # Load STACK model
        print("Loading STACK model...")
        from stack.inference import STACKPredictor
        
        predictor = STACKPredictor(
            checkpoint=str(model_path),
            genelist=str(genelist_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Load data
        print("Loading cell data...")
        prompt_adata = sc.read_h5ad(prompt_path)
        query_adata = sc.read_h5ad(query_path)
        
        print(f"Prompt cells: {prompt_adata.n_obs}")
        print(f"Query cells: {query_adata.n_obs}")
        
        # Run inference
        print(f"Running STACK inference (T={args.diffusion_steps})...")
        predictions = predictor.predict(
            prompt_adata=prompt_adata,
            query_adata=query_adata,
            T=args.diffusion_steps,
            batch_size=args.batch_size,
        )
        
        # Save predictions
        print("Saving predictions...")
        predictions.write_h5ad(output_path)
        
        # Upload to GCS
        print("Uploading predictions to GCS...")
        upload_to_gcs(output_path, args.output_gcs)
        
    print("STACK inference completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### 12.4 GCP Batch Service Client

```python
# backend/services/batch.py
"""GCP Batch client for STACK inference jobs."""

import asyncio
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from google.cloud import batch_v1
from google.cloud.batch_v1 import types


@dataclass
class BatchJobConfig:
    """Configuration for a Batch job."""
    run_id: str
    prompt_gcs_path: str
    query_gcs_path: str
    output_gcs_path: str
    model_gcs_path: str
    genelist_gcs_path: str
    diffusion_steps: int = 5
    batch_size: int = 32


class BatchClient:
    """Client for submitting and monitoring GCP Batch jobs."""
    
    def __init__(self, config: BatchConfig):
        """
        Initialize Batch client.
        
        Args:
            config: Batch configuration from Dynaconf
        """
        self.config = config
        self.client = batch_v1.BatchServiceAsyncClient()
        self.project = config.project_id
        self.region = config.region
    
    async def submit_inference_job(
        self,
        job_config: BatchJobConfig,
    ) -> str:
        """
        Submit a STACK inference job to GCP Batch.
        
        Args:
            job_config: Job configuration
        
        Returns:
            Job name (fully qualified)
        """
        job_id = f"haystack-{job_config.run_id}-{uuid4().hex[:8]}"
        
        # Build container command
        container_args = [
            f"--prompt-gcs={job_config.prompt_gcs_path}",
            f"--query-gcs={job_config.query_gcs_path}",
            f"--output-gcs={job_config.output_gcs_path}",
            f"--model-gcs={job_config.model_gcs_path}",
            f"--genelist-gcs={job_config.genelist_gcs_path}",
            f"--diffusion-steps={job_config.diffusion_steps}",
            f"--batch-size={job_config.batch_size}",
        ]
        
        # Define the container
        container = types.Runnable.Container(
            image_uri=self.config.container_image,
            commands=container_args,
        )
        
        # Define the task
        task = types.TaskSpec(
            runnables=[types.Runnable(container=container)],
            compute_resource=types.ComputeResource(
                cpu_milli=4000,  # 4 vCPUs
                memory_mib=85000,  # 85GB RAM
            ),
            max_retry_count=0,  # No automatic retries - let agent decide
            max_run_duration=f"{self.config.job_timeout_seconds}s",
        )
        
        # Define allocation policy with GPU
        allocation_policy = types.AllocationPolicy(
            instances=[
                types.AllocationPolicy.InstancePolicyOrTemplate(
                    policy=types.AllocationPolicy.InstancePolicy(
                        machine_type=self.config.machine_type,
                        accelerators=[
                            types.AllocationPolicy.Accelerator(
                                type_=self.config.accelerator_type,
                                count=self.config.accelerator_count,
                            )
                        ],
                        boot_disk=types.AllocationPolicy.Disk(
                            size_gb=self.config.boot_disk_size_gb,
                        ),
                    )
                )
            ],
            location=types.AllocationPolicy.LocationPolicy(
                allowed_locations=[f"regions/{self.region}"],
            ),
        )
        
        # Create the job
        job = types.Job(
            task_groups=[
                types.TaskGroup(
                    task_spec=task,
                    task_count=1,
                    parallelism=1,
                )
            ],
            allocation_policy=allocation_policy,
            logs_policy=types.LogsPolicy(
                destination=types.LogsPolicy.Destination.CLOUD_LOGGING,
            ),
            labels={
                "app": "haystack",
                "run-id": job_config.run_id,
            },
        )
        
        # Submit job
        request = types.CreateJobRequest(
            parent=f"projects/{self.project}/locations/{self.region}",
            job_id=job_id,
            job=job,
        )
        
        response = await self.client.create_job(request=request)
        return response.name
    
    async def get_job_status(self, job_name: str) -> tuple[str, Optional[str]]:
        """
        Get the status of a Batch job.
        
        Args:
            job_name: Fully qualified job name
        
        Returns:
            Tuple of (status, error_message)
            Status is one of: QUEUED, SCHEDULED, RUNNING, SUCCEEDED, FAILED
        """
        request = types.GetJobRequest(name=job_name)
        job = await self.client.get_job(request=request)
        
        status = types.JobStatus.State(job.status.state).name
        error_message = None
        
        if status == "FAILED":
            # Extract error message from status events
            for event in job.status.status_events:
                if "error" in event.description.lower():
                    error_message = event.description
                    break
            if not error_message:
                error_message = "Job failed with unknown error"
        
        return status, error_message
    
    async def wait_for_completion(
        self,
        job_name: str,
        poll_interval: int = 10,
        timeout: int = 1800,
    ) -> tuple[bool, Optional[str]]:
        """
        Wait for a job to complete, polling at regular intervals.
        
        Args:
            job_name: Fully qualified job name
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
        
        Returns:
            Tuple of (success, error_message)
        """
        elapsed = 0
        
        while elapsed < timeout:
            status, error = await self.get_job_status(job_name)
            
            if status == "SUCCEEDED":
                return True, None
            elif status == "FAILED":
                return False, error
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        return False, f"Job timed out after {timeout} seconds"
    
    async def cancel_job(self, job_name: str) -> None:
        """Cancel a running job."""
        request = types.DeleteJobRequest(name=job_name)
        await self.client.delete_job(request=request)
```

### 12.5 Cloud Run Configuration

```yaml
# cloud-run-deploy-prod.yaml

name: haystack-prod
region: us-east1
project: arc-prod
service_account: haystack-sa@arc-prod.iam.gserviceaccount.com

image:
  repository: us-east1-docker.pkg.dev/arc-prod/haystack/haystack
  tag: latest

env_vars:
  - DYNACONF=prod
  - FRONTEND_OUT_DIR=/app/frontend/out
  - PORT=8080

secrets:
  - ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest
  - OPENAI_API_KEY=OPENAI_API_KEY:latest

scaling:
  min_instances: 1
  max_instances: 5
  concurrency: 80

resources:
  cpu: 4
  memory: 4Gi
  timeout: 1800s  # 30 minutes for long-running analysis

vpc_access:
  connector: projects/arc-prod/locations/us-east1/connectors/haystack-vpc

cloudsql:
  instances:
    - arc-prod:us-east1:haystack-prod

iap:
  enabled: true
```

### 12.6 GCP Batch Setup

```bash
# Enable required APIs
gcloud services enable batch.googleapis.com
gcloud services enable compute.googleapis.com

# Create service account for Batch jobs
gcloud iam service-accounts create haystack-batch-sa \
    --display-name="HAYSTACK Batch Service Account"

# Grant permissions to Batch service account
# GCS access for reading inputs and writing outputs
gcloud projects add-iam-policy-binding arc-prod \
    --member="serviceAccount:haystack-batch-sa@arc-prod.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Batch job execution
gcloud projects add-iam-policy-binding arc-prod \
    --member="serviceAccount:haystack-batch-sa@arc-prod.iam.gserviceaccount.com" \
    --role="roles/batch.jobsEditor"

# Logging
gcloud projects add-iam-policy-binding arc-prod \
    --member="serviceAccount:haystack-batch-sa@arc-prod.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# Grant Cloud Run service account permission to submit Batch jobs
gcloud projects add-iam-policy-binding arc-prod \
    --member="serviceAccount:haystack-sa@arc-prod.iam.gserviceaccount.com" \
    --role="roles/batch.jobsEditor"

# Grant Cloud Run service account permission to use Batch service account
gcloud iam service-accounts add-iam-policy-binding \
    haystack-batch-sa@arc-prod.iam.gserviceaccount.com \
    --member="serviceAccount:haystack-sa@arc-prod.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Request GPU quota (if needed)
# Navigate to: https://console.cloud.google.com/iam-admin/quotas
# Search for "NVIDIA A100" in us-east1 and request increase
```

### 12.7 Build and Push STACK Inference Container

```bash
# Build STACK inference container
docker build -f Dockerfile.stack -t stack-inference:latest .

# Tag for Artifact Registry
docker tag stack-inference:latest \
    us-east1-docker.pkg.dev/arc-prod/haystack/stack-inference:latest

# Push to Artifact Registry
docker push us-east1-docker.pkg.dev/arc-prod/haystack/stack-inference:latest
```

### 12.8 GitHub Actions CI/CD

```yaml
# .github/workflows/deploy-prod.yml

name: Deploy to Production

on:
  push:
    branches: [main]

env:
  REGISTRY: us-east1-docker.pkg.dev
  PROJECT_ID: arc-prod

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Configure Docker
        run: gcloud auth configure-docker us-east1-docker.pkg.dev
      
      - name: Build and push web app
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/haystack/haystack:latest
      
      - name: Build and push STACK inference
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.stack
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/haystack/stack-inference:latest
      
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: haystack-prod
          image: ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/haystack/haystack:latest
          region: us-east1
```

---

## Related Specs

- `specification/architecture.md`
- `specification/configuration.md`
