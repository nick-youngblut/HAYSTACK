# HAYSTACK

**H**euristic **A**gent for **Y**ielding **S**TACK-**T**uned **A**ssessments with **C**losed-loop **K**nowledge

*Finding the optimal prompt in a haystack of possibilities*

## Overview

HAYSTACK is an agentic AI system that improves [STACK](https://arc.net/stack) foundation model inference through iterative, knowledge-guided prompt generation and biological grounding evaluation. Given a natural language query like "How would lung fibroblasts respond to TGF-beta treatment?", HAYSTACK:

1. **Generates biologically-informed prompts** using multiple parallel strategies (mechanistic, ontological, semantic)
2. **Executes STACK inference** with selected prompt cells for in-context learning
3. **Evaluates predictions** against pathway databases, literature, and biological priors
4. **Iteratively refines** prompts based on grounding scores until convergence

The system transforms STACK from an open-loop tool (manual prompt selection → inference → interpretation) into a closed-loop optimization system where external biological knowledge guides both prompt construction and prediction evaluation.

## Architecture

HAYSTACK uses a **two-tier GCP Batch architecture** for robust, long-running agentic workflows:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Google Cloud Run (Thin API Layer)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    FastAPI Backend (port 8080)                             │ │
│  │                                                                            │ │
│  │   POST /api/v1/runs/     → Submit CPU Batch job, return run_id            │ │
│  │   GET  /api/v1/runs/{id} → Read status from Cloud SQL                     │ │
│  │   POST /api/v1/runs/{id}/cancel → Cancel Batch job                        │ │
│  │   /*                     → Static files (Next.js)                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Next.js Frontend (Static Export)                        │ │
│  │   • Query input, status polling (15s), results visualization              │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Submits Batch job
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GCP Batch — CPU Orchestrator Job                             │
│                    e2-standard-4 (~$0.13/hr)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│   • Query understanding (LLM calls)                                             │
│   • Prompt generation (LLM + DB queries)                                        │
│   • Grounding evaluation (external APIs + LLM)                                  │
│   • Iteration control, updates Cloud SQL status                                 │
│   • Submits GPU Batch jobs for STACK inference                                  │
│   • Sends email notification on completion                                      │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Submits GPU job per iteration
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GCP Batch — GPU Inference Job                                │
│                    a2-highgpu-1g / NVIDIA A100 80GB (~$3/hr)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│   • Load STACK model, run inference (T=5), write predictions to GCS            │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                             │                             │
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Cloud SQL      │         │   GCS Buckets    │         │  External APIs   │
│   (PostgreSQL    │         │   • Atlas H5ADs  │         │  • LLM Providers │
│    + pgvector)   │         │   • STACK model  │         │  • KEGG/Reactome │
│   • Run status   │         │   • Batch I/O    │         │  • SendGrid      │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

**Benefits of this architecture:**
- Runs survive browser disconnects and Cloud Run scale-downs
- Users can launch multiple parallel runs
- No Cloud Run timeout concerns (60-min limit)
- Cost-efficient: GPU only used during actual inference

## Technology Stack

### Cloud Run (API Layer)
- **FastAPI** — Thin API layer for job submission + status queries
- **asyncpg** — PostgreSQL async driver for status reads
- **GCP Batch Client** — Submits orchestrator jobs

### CPU Batch (Orchestrator)
- **LangChain + LangGraph** — Agent orchestration and tool management
- **asyncpg + pgvector** — PostgreSQL async driver with vector similarity
- **Cloud SQL Python Connector** — Secure Cloud SQL connections
- **SendGrid** — Email notifications on run completion
- **Dynaconf** — Environment-based configuration
- **Scanpy/AnnData** — Single-cell data processing

### GPU Batch (STACK Inference)
- **NVIDIA A100 80GB** — GPU for model inference
- **PyTorch + STACK** — Foundation model inference
- **Separate container** — Built from STACK repo with PyTorch base

### Frontend
- **Next.js 14+** — React framework with App Router (static export)
- **TypeScript** — Type-safe development
- **TanStack Query** — Data fetching, caching, and polling
- **Zustand** — State management
- **Tailwind CSS + Headless UI** — Styling and components

### Infrastructure
- **Google Cloud Run** — API hosting (us-east1)
- **GCP Batch (CPU)** — Orchestrator agent jobs (us-east1)
- **GCP Batch (GPU)** — STACK inference jobs (us-east1)
- **GCP Cloud SQL** — PostgreSQL 15 with pgvector extension
- **Google Cloud Storage** — Atlas data, models, results, Batch I/O
- **GCP IAP** — Authentication (provides user email)
- **SendGrid** — Email delivery
- **Secret Manager** — Secrets storage

## Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Docker** (for local database or full container testing)
- **Google Cloud SDK** (for Cloud SQL proxy and deployment)
- **UV** (recommended Python package manager)

```bash
# Install UV if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install
```

## Project Structure

```
haystack/
├── api/                         # Cloud Run API (thin layer)
│   ├── routes/
│   │   ├── runs.py              # Run management (submit, status, cancel)
│   │   ├── cells.py             # Cell browsing
│   │   └── health.py            # Health check
│   ├── services/
│   │   ├── database.py          # Cloud SQL client (read status)
│   │   └── batch.py             # Batch job submission
│   ├── main.py                  # FastAPI app
│   └── config.py                # API configuration
│
├── orchestrator/                # CPU Batch job (agent workflow)
│   ├── agents/
│   │   ├── orchestrator.py      # Main orchestrator loop
│   │   ├── query_understanding.py
│   │   ├── prompt_generation.py
│   │   └── grounding_evaluation.py
│   ├── services/
│   │   ├── database.py          # Cloud SQL client (update status)
│   │   ├── batch.py             # GPU Batch job submission
│   │   ├── email.py             # SendGrid notifications
│   │   └── biological_apis.py   # KEGG, Reactome, etc.
│   ├── tools/                   # LangChain tools
│   └── main.py                  # Batch job entrypoint
│
├── inference/                   # GPU Batch job (STACK only)
│   └── run_inference.py         # STACK inference script
│
├── shared/                      # Shared code
│   └── models/                  # Pydantic schemas
│
├── frontend/
│   ├── app/                     # Next.js App Router pages
│   ├── components/              # React components
│   ├── hooks/                   # Custom React hooks (polling)
│   ├── lib/                     # API client, utilities
│   └── ...
│
├── docker/
│   ├── Dockerfile.api           # Cloud Run container
│   ├── Dockerfile.orchestrator  # CPU Batch container
│   └── Dockerfile.inference     # GPU Batch container
│
├── specification/               # Design documents
├── scripts/                     # Build/seed scripts
├── docker-compose.yml           # Local development
├── cloudbuild.yaml              # CI/CD configuration
├── pyproject.toml               # Python dependencies
└── README.md
```

## Local Development

### 1. Clone and Setup

```bash
git clone https://github.com/arcinstitute/haystack.git
cd haystack
```

### 2. Environment Variables

Create `.env` files for local development:

**Backend (`backend/.env`):**
```bash
# Environment
ENV_FOR_DYNACONF=development

# Database (local Docker or Cloud SQL proxy)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=haystack
DATABASE_USER=haystack_dev
DATABASE_PASSWORD=dev_password

# For Cloud SQL proxy (alternative to local Docker)
# CLOUD_SQL_INSTANCE=arc-institute:us-east1:haystack-dev
# DATABASE_USE_CLOUD_SQL_CONNECTOR=true

# Google Cloud
GCP_PROJECT=arc-institute
GCS_BUCKET_ATLASES=haystack-atlases-dev
GCS_BUCKET_RESULTS=haystack-results-dev

# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# OpenAI Embeddings
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

**Frontend (`frontend/.env.local`):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Start Local Database

Option A: **Docker Compose** (recommended for local development)

```bash
# Start PostgreSQL with pgvector
docker-compose up -d db

# Verify pgvector extension
docker exec -it haystack-db psql -U haystack_dev -d haystack -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: haystack-db
    environment:
      POSTGRES_DB: haystack
      POSTGRES_USER: haystack_dev
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - haystack_pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  haystack_pgdata:
```

Option B: **Cloud SQL Proxy** (connects to dev instance)

```bash
# Authenticate
gcloud auth application-default login

# Start proxy
cloud-sql-proxy arc-institute:us-east1:haystack-dev --port 5432
```

### 4. Initialize Database Schema

```bash
cd backend

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run migrations / initialize schema
python -m scripts.init_schema

# (Optional) Seed with sample data for development
python -m scripts.seed_db --sample-size 10000
```

### 5. Start Backend

```bash
cd backend
source .venv/bin/activate

# Development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API docs available at http://localhost:8000/docs
```

### 6. Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Open http://localhost:3000
```

### 7. Run Full Stack (Docker Compose)

For testing the complete containerized setup:

```bash
# Build and run all services
docker-compose up --build

# Access at http://localhost:8080
```

## Development Workflow

### Backend Development

```bash
cd backend
source .venv/bin/activate

# Run tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Type checking
mypy .

# Linting
ruff check .
ruff format .
```

### Frontend Development

```bash
cd frontend

# Type checking
npm run type-check

# Linting
npm run lint

# Build static export (for testing production build)
npm run build
```

### Database Migrations

We use raw SQL migrations managed via scripts:

```bash
# Apply new migration
python -m scripts.migrate up

# Rollback last migration
python -m scripts.migrate down
```

### Building the Cell Index

The cell retrieval index must be built before the system can answer queries:

```bash
# Full index build (2-4 hours, requires GPU for embeddings)
python -m scripts.build_index \
  --parse-pbmc gs://haystack-atlases/parse_pbmc.h5ad \
  --openproblems gs://haystack-atlases/openproblems.h5ad \
  --tabula-sapiens gs://haystack-atlases/tabula_sapiens.h5ad \
  --output-db postgresql://...

# Development: Build with subset of data
python -m scripts.build_index \
  --sample-size 50000 \
  --output-db postgresql://localhost:5432/haystack
```

## API Documentation

When the backend is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/runs/` | Create a new prediction run |
| `GET` | `/api/v1/runs/{run_id}` | Get run status (poll every 15s) |
| `GET` | `/api/v1/runs/{run_id}/result` | Get completed run results |
| `GET` | `/api/v1/runs/` | List user's runs |
| `POST` | `/api/v1/runs/{run_id}/cancel` | Cancel a running job |
| `GET` | `/api/v1/metadata/perturbations` | List available perturbations |
| `GET` | `/api/v1/metadata/cell-types` | List available cell types |

**Note:** Status updates use polling (every 15 seconds) rather than WebSocket. Users receive an email notification when their run completes.

## Configuration

Configuration is managed via Dynaconf with environment-specific settings:

```yaml
# backend/settings.yml
default:
  app_name: haystack
  log_level: INFO
  
  # STACK inference
  stack:
    model_path: "gs://haystack-models/stack_v1.pt"
    diffusion_steps: 5
    batch_size: 32
  
  # Retrieval
  retrieval:
    max_candidates_per_strategy: 20
    top_k_final: 10
  
  # Grounding evaluation
  grounding:
    min_score_threshold: 6
    max_iterations: 3

development:
  log_level: DEBUG
  database:
    pool_size: 5
  
production:
  database:
    pool_size: 20
  stack:
    batch_size: 64
```

## Deployment

### Cloud Run Deployment

```bash
# Build and push image
gcloud builds submit --config cloudbuild.yaml

# Or manual deployment
docker build -t us-east1-docker.pkg.dev/arc-institute/haystack/haystack:latest .
docker push us-east1-docker.pkg.dev/arc-institute/haystack/haystack:latest

gcloud run deploy haystack-prod \
  --image us-east1-docker.pkg.dev/arc-institute/haystack/haystack:latest \
  --region us-east1 \
  --platform managed \
  --memory 4Gi \
  --cpu 4 \
  --timeout 1800 \
  --vpc-connector haystack-vpc \
  --set-env-vars "ENV_FOR_DYNACONF=production" \
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest"
```
