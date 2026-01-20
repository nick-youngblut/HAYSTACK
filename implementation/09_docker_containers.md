# Sprint 09: Docker Containers

**Duration**: 1 week  
**Dependencies**: Sprint 06 (Cloud Run API), Sprint 08 (Orchestrator Batch Job)  
**Goal**: Build and configure all Docker containers for the HAYSTACK system.

---

## Overview

> **Spec Reference**: `./specification/deployment.md` (Section 12.1-12.3)

This sprint implements:
- API container (Cloud Run)
- Orchestrator container (CPU Batch)
- Inference container (GPU Batch)
- Cloud Build configuration

---

## Phase 1: API Container

### Task 1.1: Create Dockerfile.api

> **Spec Reference**: `./specification/deployment.md` (Section 12.1)

- [ ] **1.1.1** Create `docker/Dockerfile.api`:

```dockerfile
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps
COPY frontend ./
RUN npm run build

# Stage 2: Python API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api /app/api
COPY shared /app/shared

# Copy frontend build
COPY --from=frontend-build /app/frontend/out /app/frontend/out

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health/ || exit 1

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **1.1.2** Create `api/requirements.txt`:

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
asyncpg>=0.29.0
google-cloud-sql-connector[asyncpg]>=1.6.0
google-cloud-storage>=2.14.0
google-cloud-batch>=0.17.0
google-cloud-secret-manager>=2.18.0
pydantic>=2.5.0
dynaconf>=3.2.0
httpx>=0.26.0
python-multipart>=0.0.6
```

---

### Task 1.2: Build and Test API Container

- [ ] **1.2.1** Build container locally:
  ```bash
  docker build -f docker/Dockerfile.api -t haystack-api:dev .
  ```

- [ ] **1.2.2** Test container with local database:
  ```bash
  docker run -p 8080:8080 \
    -e DATABASE_URL=postgresql://... \
    haystack-api:dev
  ```

- [ ] **1.2.3** Verify health endpoint responds
- [ ] **1.2.4** Verify static frontend serves correctly

---

## Phase 2: Orchestrator Container

### Task 2.1: Create Dockerfile.orchestrator

> **Spec Reference**: `./specification/deployment.md` (Section 12.2)

- [ ] **2.1.1** Create `docker/Dockerfile.orchestrator`:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY orchestrator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY orchestrator /app/orchestrator
COPY shared /app/shared

# Non-root user for security
RUN useradd -m -u 1000 haystack && chown -R haystack:haystack /app
USER haystack

CMD ["python", "-m", "orchestrator.main"]
```

- [ ] **2.1.2** Create `orchestrator/requirements.txt`:

```
# Database
asyncpg>=0.29.0
google-cloud-sql-connector[asyncpg]>=1.6.0

# Storage
google-cloud-storage>=2.14.0
google-cloud-batch>=0.17.0

# Agent Framework
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-anthropic>=0.1.0
langchain-google-genai>=0.0.6
langgraph>=0.0.20

# Bioinformatics
scanpy>=1.9.6
anndata>=0.10.0
scipy>=1.11.0
gseapy>=1.1.0

# External APIs
httpx>=0.26.0
aiohttp>=3.9.0
biopython>=1.82

# Utilities
pydantic>=2.5.0
dynaconf>=3.2.0
sendgrid>=6.11.0
tqdm>=4.66.0
```

---

### Task 2.2: Build and Test Orchestrator Container

- [ ] **2.2.1** Build container locally
- [ ] **2.2.2** Test with mocked services
- [ ] **2.2.3** Verify all dependencies install correctly

---

## Phase 3: Inference Container

### Task 3.1: Create Dockerfile.inference

> **Spec Reference**: `./specification/deployment.md` (Section 12.3)

- [ ] **3.1.1** Create `docker/Dockerfile.inference`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install STACK model
RUN git clone https://github.com/arcinstitute/STACK.git /app/stack && \
    pip install -e /app/stack

# Install additional dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage>=2.14.0 \
    scanpy>=1.9.6 \
    anndata>=0.10.0 \
    scipy>=1.11.0 \
    tqdm>=4.66.0

# Copy inference script
COPY inference /app/inference

WORKDIR /app

ENTRYPOINT ["python", "-m", "inference.run_inference"]
```

---

### Task 3.2: Build and Test Inference Container

- [ ] **3.2.1** Build container (requires NVIDIA base image)
- [ ] **3.2.2** Test STACK model loading
- [ ] **3.2.3** Test inference with sample data
- [ ] **3.2.4** Verify GPU utilization

---

## Phase 4: Cloud Build Configuration

### Task 4.1: Create Cloud Build Config

- [ ] **4.1.1** Create `cloudbuild.yaml`:

```yaml
steps:
  # Build API container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'docker/Dockerfile.api'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/api:${SHORT_SHA}'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/api:latest'
      - '.'
    id: 'build-api'

  # Build Orchestrator container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'docker/Dockerfile.orchestrator'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/orchestrator:${SHORT_SHA}'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/orchestrator:latest'
      - '.'
    id: 'build-orchestrator'

  # Build Inference container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'docker/Dockerfile.inference'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/inference:${SHORT_SHA}'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/inference:latest'
      - '.'
    id: 'build-inference'

  # Push all images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/api']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/orchestrator']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/haystack/inference']

substitutions:
  _REGION: us-east1

timeout: '1800s'  # 30 minutes

options:
  machineType: 'E2_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY
```

---

### Task 4.2: Create Artifact Registry

- [ ] **4.2.1** Create Artifact Registry repository:
  ```bash
  gcloud artifacts repositories create haystack \
    --repository-format=docker \
    --location=us-east1 \
    --description="HAYSTACK Docker images"
  ```

- [ ] **4.2.2** Configure IAM for Cloud Build

---

## Phase 5: Local Development

### Task 5.1: Create Docker Compose

- [ ] **5.1.1** Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: haystack
      POSTGRES_USER: haystack
      POSTGRES_PASSWORD: devpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U haystack"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8080:8080"
    environment:
      - HAYSTACK_ENV=development
      - DATABASE_HOST=postgres
      - DATABASE_NAME=haystack
      - DATABASE_USER=haystack
      - DATABASE_PASSWORD=devpassword
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

---

## Definition of Done

- [ ] All three containers build successfully
- [ ] API container serves frontend and API
- [ ] Orchestrator container runs workflow
- [ ] Inference container performs STACK inference
- [ ] Cloud Build config works
- [ ] Docker Compose works for local development

---

## Next Sprint

**Sprint 10: Deployment & Configuration** - Deploy to GCP and configure production environment.
