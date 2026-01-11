# HAYSTACK: Iterative Knowledge-Guided Cell Prompting System

*Finding the optimal prompt in a haystack of possibilities*

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Goals and Non-Goals](#2-system-goals-and-non-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Models](#4-data-models)
5. [Agent Specifications](#5-agent-specifications)
6. [Tool Specifications](#6-tool-specifications)
7. [Biological Database Integration](#7-biological-database-integration)
8. [Database Specification](#8-database-specification)
9. [Backend API Specification](#9-backend-api-specification)
10. [Frontend Specification](#10-frontend-specification)
11. [Configuration](#11-configuration)
12. [Deployment](#12-deployment)
13. [Output Specification](#13-output-specification)
14. [Error Handling](#14-error-handling)
15. [Testing Strategy](#15-testing-strategy)
16. [Dependencies](#16-dependencies)
17. [Future Extensions](#17-future-extensions)

---

## 1. Executive Summary

### 1.1 Purpose

HAYSTACK (**H**euristic **A**gent for **Y**ielding **S**TACK-**T**uned **A**ssessments with **C**losed-loop **K**nowledge) is an agentic AI system that improves STACK foundation model inference through iterative, knowledge-guided prompt generation and biological grounding evaluation. The name reflects the system's core function: finding the optimal cell prompt in a "haystack" of possibilities. The system implements a closed-loop optimization approach where external biological knowledge serves as both a guide for constructing effective cell prompts and a fitness function for evaluating prediction quality.

### 1.2 Key Innovation

Traditional STACK usage is open-loop: users manually select prompts, run inference, and interpret results. HAYSTACK creates a closed-loop system where:

1. **Agent-guided prompt generation**: Multiple strategies (mechanistic, ontological, semantic) are used in parallel to explore the space of possible prompts
2. **STACK inference**: Generated prompts are used for in-context cell prompting
3. **Biological grounding evaluation**: Predictions are evaluated against pathway databases, literature, and biological priors
4. **Iterative refinement**: Evaluation feedback informs the next round of prompt generation

### 1.3 Core Capabilities

- Web-based natural language query interface for perturbation prediction requests
- Multi-strategy prompt generation leveraging drug-target knowledge, cell ontologies, and vector similarity
- Automated biological grounding evaluation with integer scoring (1-10)
- Iterative refinement with configurable stopping criteria
- Polling-based status updates with email notification on completion
- Run cancellation at any time during execution
- Support for Claude, OpenAI (GPT-5.2), and Gemini language models

### 1.4 Architecture Summary

HAYSTACK is deployed as a **FastAPI backend + Next.js frontend** web application on **Google Cloud Run**, following Arc Institute's standard full-stack patterns. The database layer uses **GCP Cloud SQL (PostgreSQL + pgvector)** for unified storage of cell metadata and vector embeddings.

---

## 2. System Goals and Non-Goals

### 2.1 Goals (MVP)

| Goal | Description |
|------|-------------|
| G1 | Accept natural language queries describing perturbation prediction tasks via web interface |
| G2 | Generate biologically-informed prompts using multiple parallel strategies |
| G3 | Execute STACK inference via GCP Batch with GPU support |
| G4 | Evaluate predictions against biological knowledge bases |
| G5 | Iteratively refine prompts based on grounding evaluation |
| G6 | Provide status updates via polling and email notification on completion |
| G7 | Allow users to cancel runs at any time |
| G8 | Produce interpretable outputs (AnnData downloads, reports, logs) |
| G9 | Support multiple LLM backends (Claude, OpenAI, Gemini) |

### 2.2 Non-Goals (MVP)

| Non-Goal | Rationale |
|----------|-----------|
| NG1 | User-provided custom datasets in vector index | Simplifies MVP; use fixed atlases |
| NG2 | Multi-species support | STACK is human-only currently |
| NG3 | Fine-tuning STACK model | Out of scope; use pre-trained checkpoints |
| NG4 | Local/on-premise deployment | Cloud Run is primary deployment target |
| NG5 | Advanced user authentication beyond GCP IAP | Simplifies MVP; use existing Arc IAP patterns |
| NG6 | Pause/resume runs for user feedback | Adds complexity; defer to future version |

### 2.3 Success Criteria

1. System produces biologically grounded predictions for >70% of queries
2. Iterative refinement improves grounding scores in >50% of cases where initial score < 7
3. End-to-end latency < 5 minutes for typical queries
4. All outputs are reproducible given the same random seed and configuration
5. Users receive email notification within 1 minute of run completion
6. Users can cancel runs and see cancellation reflected within 30 seconds

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Google Cloud Run                                      │
│                         (Single Container)                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Backend (uvicorn on port 8080)                │   │
│  │                                                                          │   │
│  │   Route Priority:                                                        │   │
│  │   1. /api/v1/* → REST API Routes (runs, cells, metadata)                 │   │
│  │   2. /*        → Static Files (/app/frontend/out)                        │   │
│  │   3. 404       → index.html (SPA routing)                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                 Next.js Frontend (Static Export)                         │   │
│  │   - App Router                                                           │   │
│  │   - TypeScript                                                           │   │
│  │   - Tailwind CSS                                                         │   │
│  │   - TanStack Query (polling for status updates)                          │   │
│  │   - Zustand (state management)                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└───────────┬────────────────────┬──────────────────────┬─────────────────────────┘
            │                    │                      │                    
            ▼                    ▼                      ▼                    
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────────────┐
│  GCP Cloud SQL   │  │  GCS Bucket      │  │  GCP Batch (STACK Inference)         │
│  (PostgreSQL +   │  │  (Atlas H5AD,    │  │                                      │
│   pgvector)      │  │   Results,       │  │  ┌────────────────────────────────┐  │
│                  │  │   STACK model)   │  │  │  NVIDIA A100 80GB VM           │  │
│  • Cell metadata │  │                  │  │  │  • STACK container             │  │
│  • Embeddings    │  │  • Prompt cells  │  │  │  • Reads prompt/query from GCS │  │
│  • Run history   │  │  • Query cells   │  │  │  • Writes predictions to GCS   │  │
│                  │  │  • Predictions   │  │  └────────────────────────────────┘  │
└──────────────────┘  └──────────────────┘  └──────────────────────────────────────┘
            │                    │                      │
            └────────────────────┼──────────────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│  External APIs               │       │  LLM Providers               │
│  (Reactome, KEGG, UniProt,   │       │  (Anthropic, OpenAI, Google) │
│   PubMed, Ensembl)           │       │                              │
└──────────────────────────────┘       └──────────────────────────────┘
                                                       │
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │  SendGrid (Email)            │
                                       │  • Completion notifications  │
                                       │  • Failure alerts            │
                                       └──────────────────────────────┘
```

### 3.2 Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HAYSTACK REQUEST FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 1. USER QUERY (via Next.js frontend)                                    │    │
│  │    • Natural language input: "How would lung fibroblasts respond        │    │
│  │      to TGF-beta treatment?"                                            │    │
│  │    • User email obtained from IAP headers                               │    │
│  │    • Returns run_id immediately; frontend polls for status              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 2. ORCHESTRATOR AGENT (Background task)                                 │    │
│  │                                                                         │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │    │
│  │    │ QUERY UNDERSTANDING SUBAGENT                                    │  │    │
│  │    │  • Parse query → StructuredQuery                                │  │    │
│  │    │  • Resolve cell type (CL ontology)                              │  │    │
│  │    │  • Resolve perturbation (DrugBank, PubChem)                     │  │    │
│  │    │  • Retrieve biological priors                                   │  │    │
│  │    └─────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                         │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │    │
│  │    │ PROMPT GENERATION SUBAGENT (Parallel Strategies)                │  │    │
│  │    │                                                                 │  │    │
│  │    │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │  │    │
│  │    │  │  Direct   │ │Mechanistic│ │ Semantic  │ │ Ontology  │        │  │    │
│  │    │  │  Match    │ │   Match   │ │ (Vector)  │ │  Guided   │        │  │    │
│  │    │  └───────────┘ └───────────┘ └───────────┘ └───────────┘        │  │    │
│  │    │         │            │             │             │              │  │    │
│  │    │         └────────────┴─────────────┴─────────────┘              │  │    │
│  │    │                              │                                  │  │    │
│  │    │                              ▼                                  │  │    │
│  │    │                  Candidate Ranking & Selection                  │  │    │
│  │    └─────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 3. STACK INFERENCE (GCP Batch Job - NVIDIA A100 80GB)                   │    │
│  │                                                                         │    │
│  │    a) Orchestrator writes prompt/query cells to GCS                     │    │
│  │    b) Submits GCP Batch job with job config                             │    │
│  │    c) Polls Batch API for job status                                    │    │
│  │    d) On completion, reads predictions from GCS                         │    │
│  │    e) On failure, error returned to agent for decision                  │    │
│  │                                                                         │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │    │
│  │    │ GCP Batch Job (Separate Container)                              │  │    │
│  │    │  • Load STACK model from GCS                                    │  │    │
│  │    │  • Load prompt/query cells from GCS                             │  │    │
│  │    │  • Run STACK (Large, T=5) inference                             │  │    │
│  │    │  • Write predictions to GCS                                     │  │    │
│  │    │  • Exit with status code                                        │  │    │
│  │    └─────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 4. GROUNDING EVALUATION SUBAGENT                                        │    │
│  │    • Extract DE genes from predictions                                  │    │
│  │    • Run pathway enrichment (GO, KEGG, Reactome)                        │    │
│  │    • Check literature support via PubMed                                │    │
│  │    • Compute integer grounding score (1-10)                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 5. ITERATION CONTROL                                                    │    │
│  │    • Check cancellation: if cancelled, stop and clean up                │    │
│  │    • Check convergence: score ≥ threshold OR max_iterations reached     │    │
│  │    • If not converged: refine prompts, submit new Batch job             │    │
│  │    • If converged: proceed to output generation                         │    │
│  │    • Each iteration = separate GCP Batch job                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 6. OUTPUT GENERATION & NOTIFICATION                                     │    │
│  │    • AnnData with predictions and metadata → GCS                        │    │
│  │    • Interpretation report (Markdown/HTML)                              │    │
│  │    • JSON execution log                                                 │    │
│  │    • Update run status in database                                      │    │
│  │    • Send email notification with results link                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND POLLING FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User submits query                                                             │
│       │                                                                         │
│       ▼                                                                         │
│  POST /api/v1/runs/ ──────► Returns { run_id, status: "pending" }               │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Poll every 15 seconds:                                                 │    │
│  │  GET /api/v1/runs/{run_id}                                              │    │
│  │                                                                         │    │
│  │  Response includes:                                                     │    │
│  │  • status: pending | running | completed | failed | cancelled           │    │
│  │  • current_iteration: 1, 2, 3...                                        │    │
│  │  • current_phase: "query_analysis" | "prompt_generation" |              │    │
│  │                   "inference" | "evaluation" | "output_generation"      │    │
│  │  • grounding_scores: [7, 8, ...]  (per iteration)                       │    │
│  │  • error_message: string (if failed)                                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  When status == "completed":                                                    │
│       │                                                                         │
│       ▼                                                                         │
│  GET /api/v1/runs/{run_id}/result ──────► Returns signed GCS URLs               │
│                                                                                 │
│  User also receives email: "Your HAYSTACK run is complete. View results: ..."   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Frontend | User interface, query input, status polling, result visualization | Next.js, TypeScript, TanStack Query |
| API Layer | REST endpoints, request validation, IAP user extraction | FastAPI, Pydantic |
| Orchestrator Agent | Manages iteration loop, coordinates subagents, submits Batch jobs | LangChain, DeepAgents |
| Query Understanding | Parses queries, resolves entities, retrieves priors | LangChain tools |
| Prompt Generation | Generates and ranks prompt candidates | LangChain tools |
| Grounding Evaluation | Evaluates predictions, computes scores | LangChain tools |
| STACK Inference | GPU-accelerated model inference (separate from Cloud Run) | GCP Batch, NVIDIA A100 80GB |
| Database | Cell metadata, vector embeddings, run history | Cloud SQL (PostgreSQL + pgvector) |
| Object Storage | Atlas H5AD files, STACK model, intermediate/final results | GCS |
| Email Service | Completion/failure notifications | SendGrid |

### 3.4 GCP Batch Job Lifecycle

Each STACK inference iteration follows this lifecycle:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       GCP BATCH JOB LIFECYCLE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Cloud Run (Orchestrator)                    GCP Batch (STACK)                  │
│  ─────────────────────────                   ──────────────────                 │
│                                                                                 │
│  1. Prepare inference inputs                                                    │
│     • Extract prompt cells from atlas                                           │
│     • Extract query cells (control state)                                       │
│     • Write prompt.h5ad to GCS                                                  │
│     • Write query.h5ad to GCS                                                   │
│            │                                                                    │
│            ▼                                                                    │
│  2. Submit Batch job ───────────────────────► Job queued                        │
│     • Job config with GPU specs                                                 │
│     • Container image + args                                                    │
│     • GCS paths for I/O                                                         │
│            │                                        │                           │
│            │                                        ▼                           │
│  3. Poll job status ◄─────────────────────── Job running                        │
│     • GET /jobs/{job_id}                     • Pull container image             │
│     • Update run status in DB                • Load STACK model                 │
│     • Sleep 10s, repeat                      • Load cells from GCS              │
│            │                                 • Run forward pass (T=5)           │
│            │                                 • Write predictions.h5ad to GCS    │
│            │                                        │                           │
│            │                                        ▼                           │
│  4. Job completed ◄──────────────────────── Job succeeded/failed                │
│            │                                                                    │
│            ▼                                                                    │
│  5. Handle result                                                               │
│     • Success: Read predictions from GCS                                        │
│     • Failure: Return error to agent                                            │
│            │                                                                    │
│            ▼                                                                    │
│  6. Continue to grounding evaluation                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
``` |

### 3.6 State Management

HAYSTACK uses a combination of state management approaches:

- **Database state**: Run metadata, cell groups, and vector embeddings stored in Cloud SQL
- **Object storage state**: Atlas files, STACK checkpoints, and result artifacts stored in GCS
- **Batch job communication**: Prompt/query cells and predictions exchanged via GCS between Cloud Run and Batch jobs
- **In-memory state**: Agent execution state maintained via LangGraph checkpointing during a run
- **Frontend state**: Run status and UI state managed via Zustand stores
- **Polling state**: Frontend polls API every 15 seconds for status updates

---

## 4. Data Models

### 4.1 Core Data Models (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime


class PerturbationType(str, Enum):
    """Types of perturbations supported."""
    DRUG = "drug"
    CYTOKINE = "cytokine"
    GENETIC = "genetic"
    UNKNOWN = "unknown"


class StructuredQuery(BaseModel):
    """Parsed representation of a user query."""
    
    raw_query: str = Field(description="Original user query text")
    
    # Cell type resolution
    cell_type_query: str = Field(description="Extracted cell type from query")
    cell_type_cl_id: Optional[str] = Field(description="Resolved Cell Ontology ID")
    cell_type_synonyms: list[str] = Field(default_factory=list)
    
    # Perturbation resolution
    perturbation_query: str = Field(description="Extracted perturbation from query")
    perturbation_type: PerturbationType
    perturbation_resolved: Optional[str] = Field(description="Canonical name")
    perturbation_external_ids: dict[str, str] = Field(default_factory=dict)
    
    # Biological priors
    expected_targets: list[str] = Field(default_factory=list)
    expected_pathways: list[str] = Field(default_factory=list)
    literature_context: Optional[str] = None


class PromptCandidate(BaseModel):
    """A candidate prompt configuration."""
    
    strategy: Literal["direct", "mechanistic", "semantic", "ontology"]
    cell_group_ids: list[str] = Field(description="Selected cell groups for prompt")
    
    # Metadata for ranking
    similarity_score: Optional[float] = None
    mechanistic_score: Optional[float] = None
    ontology_distance: Optional[int] = None
    
    # Explanation
    rationale: str = Field(description="Why this prompt was selected")


class GroundingScore(BaseModel):
    """Biological grounding evaluation result."""
    
    # Component scores (1-10)
    pathway_coherence: int = Field(ge=1, le=10)
    target_activation: int = Field(ge=1, le=10)
    literature_support: int = Field(ge=1, le=10)
    network_coherence: int = Field(ge=1, le=10)
    
    # Composite score
    composite_score: int = Field(ge=1, le=10)
    
    # Details
    enriched_pathways: list[dict]
    de_genes_up: list[str]
    de_genes_down: list[str]
    literature_evidence: list[dict]
    
    # Feedback for next iteration
    improvement_suggestions: list[str]


class IterationRecord(BaseModel):
    """Record of a single iteration."""
    
    iteration_number: int
    prompt_candidates: list[PromptCandidate]
    selected_prompt: PromptCandidate
    grounding_score: GroundingScore
    duration_seconds: float
    
    # Artifacts
    prediction_gcs_path: Optional[str] = None


class HaystackRun(BaseModel):
    """Complete run record."""
    
    # Metadata
    run_id: str
    user_email: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    status: Literal["running", "completed", "failed", "cancelled"]
    
    # Configuration
    config: dict
    random_seed: int
    
    # Query
    raw_query: str
    structured_query: Optional[StructuredQuery] = None
    
    # Iterations
    iterations: list[IterationRecord] = Field(default_factory=list)
    
    # Final result
    final_score: Optional[int] = None
    termination_reason: Optional[str] = None
    
    # Output paths (GCS)
    output_anndata_path: Optional[str] = None
    output_report_path: Optional[str] = None
    output_log_path: Optional[str] = None


class RunListResponse(BaseModel):
    """Response for listing runs."""
    
    runs: list[HaystackRun]
    total: int
    page: int
    page_size: int
```

### 4.2 API Request/Response Models

```python
class CreateRunRequest(BaseModel):
    """Request to create a new HAYSTACK run."""
    
    query: str = Field(description="Natural language query", min_length=10)
    
    # Optional overrides
    max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
    score_threshold: Optional[int] = Field(default=None, ge=1, le=10)
    llm_provider: Optional[Literal["anthropic", "openai", "google_genai"]] = None
    llm_model: Optional[str] = None
    random_seed: Optional[int] = None


class RunStatusResponse(BaseModel):
    """Response with run status."""
    
    run_id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    current_iteration: int
    max_iterations: int
    current_score: Optional[int] = None
    message: str


class RunResultResponse(BaseModel):
    """Response with run results."""
    
    run_id: str
    success: bool
    grounding_score: int
    termination_reason: str
    
    # Predictions summary
    num_de_genes: int
    top_upregulated: list[str]
    top_downregulated: list[str]
    
    # Interpretation
    enriched_pathways: list[str]
    activated_tfs: list[str]
    
    # Download URLs (signed GCS URLs)
    anndata_url: str
    report_url: str
    log_url: str


class RunPhase(str, Enum):
    """Current phase of a run for status reporting."""
    PENDING = "pending"
    QUERY_ANALYSIS = "query_analysis"
    PROMPT_GENERATION = "prompt_generation"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    OUTPUT_GENERATION = "output_generation"


class RunStatusResponse(BaseModel):
    """Response model for run status polling."""
    
    run_id: str
    status: RunStatus
    current_iteration: int = 0
    max_iterations: int
    current_phase: Optional[RunPhase] = None
    grounding_scores: list[int] = Field(default_factory=list)  # Score per iteration
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    
    # User info
    user_email: str


class EmailNotification(BaseModel):
    """Email notification configuration."""
    
    recipient_email: str
    subject: str
    template: Literal["run_completed", "run_failed", "run_cancelled"]
    template_data: dict = Field(default_factory=dict)
```

### 4.3 Configuration Models

```python
class LLMConfig(BaseModel):
    """LLM backend configuration."""
    
    provider: Literal["anthropic", "openai", "google_genai"]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Provider-specific
    api_key_env_var: str = Field(
        description="Environment variable containing API key"
    )


class IterationConfig(BaseModel):
    """Iteration control configuration."""
    
    max_iterations: int = 5
    score_threshold: int = 7  # 1-10 scale
    plateau_window: int = 3  # Stop if no improvement over N iterations
    min_improvement: int = 1  # Minimum score improvement to not count as plateau


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    # Cloud SQL connection
    instance_connection_name: str
    database_name: str = "haystack"
    user: str = "haystack_app"
    
    # Connection pool
    pool_size: int = 5
    max_overflow: int = 10


class GCSConfig(BaseModel):
    """Google Cloud Storage configuration."""
    
    project_id: str
    bucket_name: str
    
    # Paths within bucket
    atlases_prefix: str = "atlases/"
    stack_model_prefix: str = "models/stack/"
    results_prefix: str = "results/"


class DatabaseAPIConfig(BaseModel):
    """Biological database API configuration."""
    
    # Retry settings
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    
    # Rate limiting
    requests_per_minute: int = 30


class HaystackConfig(BaseModel):
    """Complete HAYSTACK configuration."""
    
    # Environment
    environment: Literal["dev", "prod"] = "dev"
    debug: bool = False
    
    # Components
    llm: LLMConfig
    iteration: IterationConfig
    database: DatabaseConfig
    gcs: GCSConfig
    database_apis: DatabaseAPIConfig
    
    # STACK model
    stack_checkpoint_path: str  # GCS path
    stack_genelist_path: str    # GCS path
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
```

---

## 5. Agent Specifications

### 5.1 Orchestrator Agent

The orchestrator is the main entry point, implemented using DeepAgents with FastAPI integration.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

def create_orchestrator(config: HaystackConfig, run_id: str):
    """
    Create orchestrator agent for a HAYSTACK run.
    
    Args:
        config: HAYSTACK configuration
        run_id: Unique run identifier
    
    Returns:
        Configured DeepAgent orchestrator
    """
    # Initialize model
    model = init_chat_model(
        f"{config.llm.provider}:{config.llm.model}",
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    orchestrator = create_deep_agent(
        model=model,
        tools=[
            # Direct tools
            check_convergence_tool,
            generate_report_tool,
            run_stack_inference_tool,
            save_results_to_gcs_tool,
        ],
        subagents=[
            query_understanding_subagent,
            prompt_generation_subagent,
            grounding_evaluation_subagent,
        ],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        backend=FilesystemBackend(root_dir=f"/tmp/haystack/{run_id}"),
    )
    
    return orchestrator
```

**Orchestrator System Prompt**:
```
You are the HAYSTACK orchestrator, an AI system that improves single-cell perturbation predictions through iterative knowledge-guided prompting.

Your workflow:
1. UNDERSTAND: Parse the user's query to identify cell type, perturbation, and expected biology
2. GENERATE: Create multiple prompt candidates using different strategies
3. INFER: Run STACK inference with selected prompts
4. EVALUATE: Assess biological grounding of predictions (1-10 score)
5. DECIDE: If score ≥ threshold OR max iterations reached → finalize; else → refine and repeat

Key principles:
- Always explain your reasoning before taking actions
- Use biological knowledge to guide prompt selection
- Consider multiple strategies in parallel
- Learn from evaluation feedback to improve subsequent iterations
- Be conservative with iteration count; stop when predictions are well-grounded
```

### 5.2 Query Understanding Subagent

```python
query_understanding_subagent = create_deep_agent(
    model=model,
    tools=[
        resolve_cell_type_tool,
        resolve_perturbation_tool,
        get_drug_targets_tool,
        get_pathway_priors_tool,
        search_literature_tool,
    ],
    system_prompt=QUERY_UNDERSTANDING_PROMPT,
)
```

**System Prompt**:
```
You are a biological query understanding agent. Your job is to:

1. Extract cell type and perturbation from natural language queries
2. Resolve cell types to Cell Ontology IDs using the resolve_cell_type tool
3. Resolve perturbations to canonical names and external IDs
4. Retrieve known targets and pathway associations
5. Search literature for relevant biological context

Output a StructuredQuery with all resolved information.

Be thorough in resolving entities - this information guides prompt selection.
```

### 5.3 Prompt Generation Subagent

```python
prompt_generation_subagent = create_deep_agent(
    model=model,
    tools=[
        search_cells_by_perturbation_tool,
        search_cells_by_cell_type_tool,
        semantic_search_cells_tool,
        find_mechanistically_similar_tool,
        find_ontology_related_cells_tool,
        rank_prompt_candidates_tool,
    ],
    system_prompt=PROMPT_GENERATION_PROMPT,
)
```

**System Prompt**:
```
You are a prompt generation agent for STACK in-context learning. Your job is to select the best "prompt cells" from the available atlases.

Strategies (use all in parallel):
1. DIRECT: Find exact matches for cell type and perturbation
2. MECHANISTIC: Find perturbations sharing targets/pathways with query
3. SEMANTIC: Use vector similarity for related perturbations
4. ONTOLOGY: Find related cell types via CL ontology hierarchy

After generating candidates from each strategy, rank them by:
- Biological relevance to the query
- Data quality (cell count, coverage)
- Diversity of selected cells

Return the top-ranked PromptCandidate with rationale.
```

### 5.4 Grounding Evaluation Subagent

```python
grounding_evaluation_subagent = create_deep_agent(
    model=model,
    tools=[
        extract_de_genes_tool,
        run_pathway_enrichment_tool,
        check_target_activation_tool,
        search_literature_evidence_tool,
        build_gene_network_tool,
        compute_grounding_score_tool,
    ],
    system_prompt=GROUNDING_EVALUATION_PROMPT,
)
```

**System Prompt**:
```
You are a biological grounding evaluation agent. Your job is to assess how well STACK predictions align with biological knowledge.

Evaluation criteria (each scored 1-10):
1. PATHWAY COHERENCE: Do enriched pathways match expected biology?
2. TARGET ACTIVATION: Are known targets differentially expressed correctly?
3. LITERATURE SUPPORT: Do predictions have published evidence?
4. NETWORK COHERENCE: Do DE genes form connected functional modules?

Compute a composite score and provide actionable feedback for improvement.

Be critical but fair - novel predictions that make biological sense should not be penalized.
```

---

## 6. Tool Specifications

### 6.1 Database Query Tools

```python
@tool
async def search_cells_by_perturbation(
    perturbation_name: str,
    cell_type_cl_id: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Search for cells by perturbation name with optional filters.
    
    Args:
        perturbation_name: Canonical perturbation name
        cell_type_cl_id: Optional Cell Ontology ID filter
        dataset: Optional dataset filter (parse_pbmc, openproblems, tabula_sapiens)
        limit: Maximum results to return
    
    Returns:
        List of cell groups matching criteria
    """
    ...


@tool
async def semantic_search_cells(
    query_text: str,
    search_type: Literal["perturbation", "cell_type"],
    top_k: int = 50,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """
    Vector similarity search for cells using text embeddings.
    
    Args:
        query_text: Natural language description
        search_type: Which embedding to search
        top_k: Number of results
        similarity_threshold: Minimum cosine similarity
    
    Returns:
        List of cell groups with similarity scores
    """
    ...


@tool
async def find_ontology_related_cells(
    cell_type_cl_id: str,
    max_distance: int = 2,
    perturbation_filter: Optional[str] = None,
) -> list[dict]:
    """
    Find cells with related cell types via Cell Ontology hierarchy.
    
    Args:
        cell_type_cl_id: Query cell type CL ID
        max_distance: Maximum ontology distance (parent/child levels)
        perturbation_filter: Optional perturbation filter
    
    Returns:
        List of cell groups with ontology distance
    """
    ...
```

### 6.2 Drug-Target Knowledge Tools

```python
@tool
async def get_drug_targets(
    perturbation: str,
    perturbation_type: str,
) -> dict:
    """
    Retrieve known targets for a perturbation.
    
    Args:
        perturbation: Name of drug/cytokine/gene
        perturbation_type: One of 'drug', 'cytokine', 'genetic'
    
    Returns:
        Dictionary with targets, target_types, sources, confidence
    
    Databases:
        - KEGG DRUG (for drugs)
        - UniProt (for receptors/binding partners)
        - Reactome (for signaling components)
    """
    ...


@tool
async def get_pathway_memberships(genes: list[str]) -> dict:
    """
    Get pathway memberships for a list of genes.
    
    Args:
        genes: List of gene symbols
    
    Returns:
        Dictionary with kegg_pathways, reactome_pathways, go_terms
    """
    ...


@tool
async def find_mechanistically_similar_perturbations(
    target_genes: list[str],
    pathways: list[str],
    available_perturbations: list[str],
) -> list[dict]:
    """
    Find perturbations sharing targets or pathways with query.
    
    Args:
        target_genes: Known target genes
        pathways: Associated pathways
        available_perturbations: Perturbations in atlas
    
    Returns:
        List of similar perturbations with overlap scores
    """
    ...
```

### 6.3 STACK Inference Tools

```python
@tool
async def run_stack_inference(
    prompt_cell_group_ids: list[str],
    query_cell_group_ids: list[str],
    run_id: str,
    iteration: int,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run STACK inference with selected prompt and query cells.
    
    Args:
        prompt_cell_group_ids: Cell group IDs for prompt
        query_cell_group_ids: Cell group IDs for query
        run_id: Current run ID
        iteration: Current iteration number
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with:
        - status: 'completed' or 'failed'
        - prediction_path: GCS path to predictions
        - duration_seconds: Inference time
        - metrics: Model quality metrics
    
    Implementation:
        Runs STACK Large (T=5) using asyncio executor
    """
    ...


@tool
async def extract_de_genes(
    prediction_path: str,
    control_path: str,
    lfc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
) -> list[dict]:
    """
    Extract differentially expressed genes from predictions.
    
    Args:
        prediction_path: GCS path to prediction AnnData
        control_path: GCS path to control AnnData
        lfc_threshold: Minimum log2 fold change
        pval_threshold: Maximum adjusted p-value
    
    Returns:
        List of DE genes with statistics
    """
    ...
```

### 6.4 Enrichment and Evaluation Tools

```python
@tool
async def run_pathway_enrichment(
    genes: list[str],
    background_genes: Optional[list[str]] = None,
    databases: list[str] = ["GO_BP", "KEGG", "Reactome"],
) -> dict:
    """
    Run pathway enrichment analysis using gseapy.
    
    Args:
        genes: List of gene symbols
        background_genes: Optional background gene set
        databases: Databases to query
    
    Returns:
        Dictionary with enrichment results per database
    """
    ...


@tool
async def compute_grounding_score(
    de_genes: list[dict],
    expected_targets: list[str],
    expected_pathways: list[str],
    enrichment_results: dict,
    literature_evidence: list[dict],
) -> GroundingScore:
    """
    Compute composite biological grounding score.
    
    Args:
        de_genes: Differentially expressed genes
        expected_targets: Known perturbation targets
        expected_pathways: Expected affected pathways
        enrichment_results: Pathway enrichment results
        literature_evidence: PubMed evidence
    
    Returns:
        GroundingScore with component and composite scores
    """
    ...
```

---

## 7. Biological Database Integration

### 7.1 External API Configuration

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx


class BiologicalDatabaseClient:
    """Async client for biological database APIs."""
    
    def __init__(self, config: DatabaseAPIConfig):
        """
        Initialize database client.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AsyncRateLimiter(config.requests_per_minute)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
    )
    async def query_kegg(self, endpoint: str, params: dict) -> dict:
        """Query KEGG API with retry logic."""
        async with self._rate_limiter:
            response = await self.client.get(
                f"https://rest.kegg.jp/{endpoint}",
                params=params,
            )
            response.raise_for_status()
            return self._parse_kegg_response(response.text)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
    )
    async def query_reactome(self, endpoint: str, params: dict) -> dict:
        """Query Reactome API with retry logic."""
        async with self._rate_limiter:
            response = await self.client.get(
                f"https://reactome.org/ContentService/{endpoint}",
                params=params,
            )
            response.raise_for_status()
            return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

### 7.2 Supported Databases

| Database | Purpose | Endpoint |
|----------|---------|----------|
| KEGG | Pathways, drug targets | https://rest.kegg.jp |
| Reactome | Pathway analysis | https://reactome.org/ContentService |
| UniProt | Protein information | https://rest.uniprot.org |
| PubMed | Literature search | https://eutils.ncbi.nlm.nih.gov |
| Cell Ontology | Cell type hierarchy | Local OBO file |
| Gene Ontology | GO terms | https://api.geneontology.org |

---

## 8. Database Specification

### 8.1 Cloud SQL Setup

HAYSTACK uses **GCP Cloud SQL (PostgreSQL 15 + pgvector)** for unified storage of cell metadata and vector embeddings.

**Instance Configuration:**
- Instance type: `db-custom-4-15360` (4 vCPU, 15 GB RAM)
- Storage: 100 GB SSD with auto-resize
- Region: `us-east1` (same as Cloud Run and Batch)
- High availability: Enabled for production
- Private IP: Enabled via VPC connector

### 8.2 Database Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main cells table
CREATE TABLE cells (
    id SERIAL PRIMARY KEY,
    
    -- Identifiers
    cell_index INT NOT NULL,
    group_id VARCHAR(64) NOT NULL,
    dataset VARCHAR(32) NOT NULL,
    
    -- Cell type (harmonized)
    cell_type_original VARCHAR(256),
    cell_type_cl_id VARCHAR(32),
    cell_type_name VARCHAR(256),
    
    -- Perturbation (harmonized)
    perturbation_original VARCHAR(256),
    perturbation_name VARCHAR(256),
    perturbation_type VARCHAR(32),
    is_control BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Additional metadata
    tissue_original VARCHAR(256),
    tissue_uberon_id VARCHAR(32),
    donor_id VARCHAR(64),
    
    -- External IDs (JSONB for flexibility)
    perturbation_external_ids JSONB DEFAULT '{}',
    perturbation_targets TEXT[],
    perturbation_pathways TEXT[],
    
    -- Quality metrics
    n_genes INT,
    total_counts FLOAT,
    
    -- Text embeddings for semantic search (text-embedding-3-large, 1536 dim)
    perturbation_embedding vector(1536),
    cell_type_embedding vector(1536),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cell groups table (aggregated view)
CREATE TABLE cell_groups (
    group_id VARCHAR(64) PRIMARY KEY,
    dataset VARCHAR(32) NOT NULL,
    perturbation_name VARCHAR(256),
    cell_type_cl_id VARCHAR(32),
    donor_id VARCHAR(64),
    
    n_cells INT NOT NULL,
    cell_indices INT[] NOT NULL,
    
    mean_n_genes FLOAT,
    mean_total_counts FLOAT,
    
    has_control BOOLEAN DEFAULT FALSE,
    control_group_id VARCHAR(64),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Perturbation lookup table
CREATE TABLE perturbations (
    perturbation_name VARCHAR(256) PRIMARY KEY,
    perturbation_type VARCHAR(32),
    external_ids JSONB DEFAULT '{}',
    targets TEXT[],
    pathways TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    cell_types_present TEXT[]
);

-- Cell type lookup table
CREATE TABLE cell_types (
    cell_type_cl_id VARCHAR(32) PRIMARY KEY,
    cell_type_name VARCHAR(256) NOT NULL,
    lineage TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    perturbations_present TEXT[]
);

-- Synonym table for fuzzy matching
CREATE TABLE synonyms (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(256) NOT NULL,
    synonym VARCHAR(256) NOT NULL,
    entity_type VARCHAR(32) NOT NULL  -- 'perturbation' or 'cell_type'
);

-- Run history table
CREATE TABLE runs (
    run_id VARCHAR(64) PRIMARY KEY,
    user_email VARCHAR(256),
    status VARCHAR(32) NOT NULL,
    
    raw_query TEXT NOT NULL,
    structured_query JSONB,
    config JSONB NOT NULL,
    random_seed INT,
    
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    
    iterations JSONB DEFAULT '[]',
    final_score INT,
    termination_reason VARCHAR(256),
    
    output_anndata_path VARCHAR(512),
    output_report_path VARCHAR(512),
    output_log_path VARCHAR(512),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX idx_cells_dataset ON cells(dataset);
CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
CREATE INDEX idx_cells_tissue ON cells(tissue_uberon_id);
CREATE INDEX idx_cells_is_control ON cells(is_control);
CREATE INDEX idx_cells_group ON cells(group_id);
CREATE INDEX idx_cells_donor ON cells(donor_id);

-- HNSW vector indexes for similarity search
CREATE INDEX idx_cells_perturbation_embedding ON cells 
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_cell_type_embedding ON cells 
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index on synonym table
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
CREATE INDEX idx_synonyms_type ON synonyms(entity_type);

-- Index on runs table
CREATE INDEX idx_runs_user ON runs(user_email);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at DESC);
```

### 8.3 Database Roles

```sql
-- Application role (read-write for runs, read-only for cells)
CREATE ROLE haystack_app WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_app;
GRANT USAGE ON SCHEMA public TO haystack_app;
GRANT SELECT ON cells, cell_groups, perturbations, cell_types, synonyms TO haystack_app;
GRANT SELECT, INSERT, UPDATE ON runs TO haystack_app;
GRANT USAGE, SELECT ON SEQUENCE runs_run_id_seq TO haystack_app;

-- Agent role (read-only for all tables)
CREATE ROLE haystack_agent WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_agent;
GRANT USAGE ON SCHEMA public TO haystack_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO haystack_agent;
ALTER ROLE haystack_agent SET statement_timeout = '30s';
ALTER ROLE haystack_agent SET work_mem = '256MB';

-- Admin role (full access)
CREATE ROLE haystack_admin WITH LOGIN PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE haystack TO haystack_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO haystack_admin;
```

### 8.4 Python Database Client

```python
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional
from google.cloud.sql.connector import Connector


class HaystackDatabase:
    """Async database client for HAYSTACK using Cloud SQL."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._connector = Connector()
    
    async def _get_connection(self):
        """Get connection using Cloud SQL Python Connector."""
        return await self._connector.connect_async(
            self.config.instance_connection_name,
            "asyncpg",
            user=self.config.user,
            db=self.config.database_name,
            enable_iam_auth=True,
        )
    
    async def connect(self):
        """Initialize connection pool."""
        self._pool = await asyncpg.create_pool(
            min_size=2,
            max_size=self.config.pool_size,
            max_inactive_connection_lifetime=300,
            setup=self._get_connection,
        )
    
    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
        self._connector.close()
    
    @asynccontextmanager
    async def connection(self):
        """Get a database connection from pool."""
        async with self._pool.acquire() as conn:
            yield conn
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        max_rows: int = 1000,
    ) -> list[dict]:
        """
        Execute a read-only SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            max_rows: Maximum rows to return
        
        Returns:
            List of result dictionaries
        """
        async with self.connection() as conn:
            if params:
                rows = await conn.fetch(sql, *params, timeout=30)
            else:
                rows = await conn.fetch(sql, timeout=30)
            return [dict(row) for row in rows[:max_rows]]
    
    async def semantic_search(
        self,
        query_embedding: list[float],
        search_type: str,
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Vector similarity search using pgvector.
        
        Args:
            query_embedding: Query vector (1536 dim)
            search_type: 'perturbation' or 'cell_type'
            top_k: Number of results
            filters: Optional SQL filters
        
        Returns:
            List of results with similarity scores
        """
        embedding_col = f"{search_type}_embedding"
        
        sql = f"""
            SELECT 
                group_id,
                perturbation_name,
                cell_type_cl_id,
                cell_type_name,
                dataset,
                1 - ({embedding_col} <=> $1::vector) as similarity
            FROM cells
            WHERE {embedding_col} IS NOT NULL
        """
        
        if filters:
            for key, value in filters.items():
                sql += f" AND {key} = ${len(filters) + 1}"
        
        sql += f"""
            ORDER BY {embedding_col} <=> $1::vector
            LIMIT {top_k}
        """
        
        params = [query_embedding]
        if filters:
            params.extend(filters.values())
        
        async with self.connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]
```

---

## 9. Backend API Specification

### 9.1 Project Structure

```
backend/
├── __init__.py
├── main.py                 # FastAPI app factory
├── config.py               # Dynaconf settings
├── context.py              # Dependency injection context
├── dependencies.py         # FastAPI dependencies
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       ├── runs.py         # /api/v1/runs endpoints
│       ├── cells.py        # /api/v1/cells endpoints
│       ├── metadata.py     # /api/v1/metadata endpoints
│       └── health.py       # Health check
├── models/
│   ├── __init__.py
│   ├── runs.py             # Run-related Pydantic models
│   ├── cells.py            # Cell-related models
│   └── notifications.py    # Email notification models
├── services/
│   ├── __init__.py
│   ├── database.py         # HaystackDatabase client
│   ├── gcs.py              # GCS operations
│   ├── biological_apis.py  # External API clients
│   ├── stack_inference.py  # STACK model wrapper
│   ├── batch.py            # GCP Batch client
│   └── email.py            # SendGrid email notifications
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py     # Main orchestrator
│   ├── query_understanding.py
│   ├── prompt_generation.py
│   └── grounding_evaluation.py
├── tools/
│   ├── __init__.py
│   ├── database_tools.py
│   ├── knowledge_tools.py
│   ├── inference_tools.py
│   └── enrichment_tools.py
├── workers/
│   └── run_worker.py       # Background run execution
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api/
    └── test_services/
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

from backend.config import settings
from backend.api.routes import runs, cells, metadata, health
from backend.services.database import database
from backend.services.gcs import gcs_service
from backend.services.email import email_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await database.connect()
    await gcs_service.initialize()
    email_service.initialize()
    yield
    # Shutdown
    await database.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
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
    app.include_router(cells.router, prefix="/api/v1/cells", tags=["cells"])
    app.include_router(metadata.router, prefix="/api/v1/metadata", tags=["metadata"])
    
    # Serve static frontend (production)
    frontend_dir = os.environ.get("FRONTEND_OUT_DIR", "/app/frontend/out")
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    
    return app


app = create_app()
```

### 9.3 API Endpoints

```python
# backend/api/routes/runs.py
"""Run management endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Optional

from backend.models.runs import (
    CreateRunRequest,
    RunStatusResponse,
    RunResultResponse,
    RunListResponse,
)
from backend.services.database import database
from backend.agents.orchestrator import run_haystack_pipeline
from backend.dependencies import get_current_user

router = APIRouter()


@router.post("/", response_model=RunStatusResponse)
async def create_run(
    request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    user_email: str = Depends(get_current_user),
):
    """
    Create a new HAYSTACK run.
    
    Starts the iterative optimization pipeline in the background.
    Returns immediately with run_id for status polling.
    """
    run_id = generate_run_id()
    
    # Create run record
    await database.create_run(
        run_id=run_id,
        user_email=user_email,
        query=request.query,
        config=request.dict(),
    )
    
    # Start pipeline in background
    background_tasks.add_task(
        run_haystack_pipeline,
        run_id=run_id,
        query=request.query,
        config=request.dict(),
    )
    
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
    
    if run["status"] != "running":
        raise HTTPException(400, f"Run {run_id} is not running")
    
    await database.update_run_status(run_id, "cancelled")
    return {"message": f"Run {run_id} cancelled"}
```

### 9.4 IAP User Extraction

```python
# backend/dependencies.py
"""FastAPI dependencies including IAP user extraction."""

from fastapi import Request, HTTPException


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

### 9.5 Email Notification Service

```python
# backend/services/email.py
"""Email notification service using SendGrid."""

import logging
from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from backend.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending email notifications."""
    
    def __init__(self):
        self.client: Optional[SendGridAPIClient] = None
        self.from_email = "haystack@arc.institute"
        self.base_url = settings.frontend_url
    
    def initialize(self):
        """Initialize SendGrid client."""
        api_key = settings.sendgrid_api_key
        if api_key:
            self.client = SendGridAPIClient(api_key)
            logger.info("SendGrid client initialized")
        else:
            logger.warning("SENDGRID_API_KEY not set, emails disabled")
    
    async def send_run_completed(
        self,
        recipient_email: str,
        run_id: str,
        query: str,
        grounding_score: int,
    ) -> bool:
        """
        Send email notification when a run completes successfully.
        
        Args:
            recipient_email: User's email address
            run_id: HAYSTACK run ID
            query: Original user query
            grounding_score: Final grounding score
        
        Returns:
            True if email sent successfully
        """
        if not self.client:
            logger.warning(f"Email disabled, skipping notification for run {run_id}")
            return False
        
        results_url = f"{self.base_url}/runs/{run_id}"
        
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

### 9.6 Run Worker with Email Notifications

```python
# backend/workers/run_worker.py
"""Background worker for executing HAYSTACK runs."""

import logging
from datetime import datetime

from backend.services.database import database
from backend.services.email import email_service
from backend.services.batch import batch_client
from backend.agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)


async def run_haystack_pipeline(
    run_id: str,
    query: str,
    config: dict,
):
    """
    Execute the HAYSTACK pipeline in the background.
    
    Updates run status in database and sends email notification on completion.
    
    Args:
        run_id: Unique run identifier
        query: User's natural language query
        config: Run configuration
    """
    try:
        # Update status to running
        await database.update_run(
            run_id=run_id,
            status="running",
            current_phase="query_analysis",
            updated_at=datetime.utcnow(),
        )
        
        # Get run record for user email
        run = await database.get_run(run_id)
        user_email = run["user_email"]
        
        # Initialize orchestrator agent
        orchestrator = OrchestratorAgent(config=config)
        
        # Execute iterative pipeline
        iteration = 0
        max_iterations = config.get("max_iterations", 5)
        score_threshold = config.get("score_threshold", 7)
        
        while iteration < max_iterations:
            # Check for cancellation
            run = await database.get_run(run_id)
            if run["status"] == "cancelled":
                logger.info(f"Run {run_id} was cancelled")
                await email_service.send_run_cancelled(
                    recipient_email=user_email,
                    run_id=run_id,
                    query=query,
                )
                return
            
            iteration += 1
            
            # Update phase: prompt generation
            await database.update_run(
                run_id=run_id,
                current_phase="prompt_generation",
                current_iteration=iteration,
                updated_at=datetime.utcnow(),
            )
            
            # Generate prompts
            prompt_cells = await orchestrator.generate_prompts(query, iteration)
            
            # Update phase: inference
            await database.update_run(
                run_id=run_id,
                current_phase="inference",
                updated_at=datetime.utcnow(),
            )
            
            # Submit Batch job and wait for completion
            predictions = await orchestrator.run_inference(
                run_id=run_id,
                iteration=iteration,
                prompt_cells=prompt_cells,
            )
            
            # Check for cancellation again after long inference
            run = await database.get_run(run_id)
            if run["status"] == "cancelled":
                await email_service.send_run_cancelled(
                    recipient_email=user_email,
                    run_id=run_id,
                    query=query,
                )
                return
            
            # Update phase: evaluation
            await database.update_run(
                run_id=run_id,
                current_phase="evaluation",
                updated_at=datetime.utcnow(),
            )
            
            # Evaluate predictions
            grounding_result = await orchestrator.evaluate_predictions(predictions)
            score = grounding_result.score
            
            # Update grounding scores
            scores = run.get("grounding_scores", [])
            scores.append(score)
            await database.update_run(
                run_id=run_id,
                grounding_scores=scores,
                updated_at=datetime.utcnow(),
            )
            
            # Check convergence
            if score >= score_threshold:
                logger.info(f"Run {run_id} converged with score {score}")
                break
        
        # Update phase: output generation
        await database.update_run(
            run_id=run_id,
            current_phase="output_generation",
            updated_at=datetime.utcnow(),
        )
        
        # Generate final outputs
        output_paths = await orchestrator.generate_outputs(run_id)
        
        # Mark as completed
        final_score = scores[-1] if scores else 0
        await database.update_run(
            run_id=run_id,
            status="completed",
            current_phase=None,
            final_score=final_score,
            output_anndata_path=output_paths["anndata"],
            output_report_path=output_paths["report"],
            output_log_path=output_paths["log"],
            completed_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        # Send completion email
        await email_service.send_run_completed(
            recipient_email=user_email,
            run_id=run_id,
            query=query,
            grounding_score=final_score,
        )
        
        logger.info(f"Run {run_id} completed successfully with score {final_score}")
        
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}")
        
        # Update status to failed
        await database.update_run(
            run_id=run_id,
            status="failed",
            error_message=str(e),
            updated_at=datetime.utcnow(),
        )
        
        # Send failure email
        run = await database.get_run(run_id)
        await email_service.send_run_failed(
            recipient_email=run["user_email"],
            run_id=run_id,
            query=query,
            error_message=str(e),
        )
```

---

## 10. Frontend Specification

### 10.1 Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout with Providers
│   ├── page.tsx                # Home/dashboard
│   ├── globals.css             # Tailwind imports
│   ├── runs/
│   │   ├── page.tsx            # /runs - list runs
│   │   ├── new/page.tsx        # /runs/new - create run
│   │   └── [id]/page.tsx       # /runs/[id] - run detail
│   └── explore/
│       └── page.tsx            # /explore - browse cells
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── PageLayout.tsx
│   ├── runs/
│   │   ├── RunForm.tsx         # Query input form
│   │   ├── RunStatus.tsx       # Status display with polling
│   │   ├── RunResults.tsx      # Results visualization
│   │   └── RunHistory.tsx      # Past runs table
│   ├── explore/
│   │   ├── CellBrowser.tsx     # Atlas browser
│   │   └── SearchFilters.tsx
│   ├── ui/
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ProgressBar.tsx
│   └── Providers.tsx           # React Query + Toaster
├── hooks/
│   ├── queries/
│   │   ├── useRuns.ts          # Run queries with polling
│   │   ├── useCells.ts
│   │   └── useMetadata.ts
│   └── useRunPolling.ts        # Status polling hook
├── stores/
│   ├── runStore.ts             # Current run state
│   └── uiStore.ts              # UI state
├── lib/
│   ├── api/
│   │   ├── client.ts
│   │   ├── runs.ts
│   │   └── cells.ts
│   └── query-client.ts
├── types/
│   ├── api.ts
│   └── runs.ts
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── next.config.js
```

### 10.2 Key Components

```typescript
// components/runs/RunForm.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { createRun } from "@/lib/api/runs";
import { Button } from "@/components/ui/Button";

export function RunForm() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  
  const mutation = useMutation({
    mutationFn: createRun,
    onSuccess: (data) => {
      router.push(`/runs/${data.run_id}`);
    },
  });
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate({ query });
  };
  
  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">
          Describe your perturbation prediction task
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="How would lung fibroblasts respond to TGF-beta treatment?"
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          rows={4}
        />
      </div>
      
      <Button
        type="submit"
        disabled={mutation.isPending || query.length < 10}
      >
        {mutation.isPending ? "Starting..." : "Start Analysis"}
      </Button>
    </form>
  );
}
```

```typescript
// components/runs/RunProgress.tsx
"use client";

import { useRunProgress } from "@/hooks/useRunProgress";
import { ProgressBar } from "@/components/ui/ProgressBar";

interface RunProgressProps {
  runId: string;
}

export function RunProgress({ runId }: RunProgressProps) {
  const { status, currentIteration, maxIterations, currentScore, messages } = 
    useRunProgress(runId);
  
  const progress = (currentIteration / maxIterations) * 100;
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <span>Iteration {currentIteration} of {maxIterations}</span>
        {currentScore && <span>Score: {currentScore}/10</span>}
      </div>
      
      <ProgressBar value={progress} />
      
      <div className="space-y-2">
        {messages.map((msg, i) => (
          <div key={i} className="text-sm text-gray-600">
            {msg.type === "iteration_start" && (
              <span>🔄 Starting iteration {msg.data?.iteration}...</span>
            )}
            {msg.type === "progress" && (
              <span>📊 {msg.data?.message}</span>
            )}
            {msg.type === "iteration_end" && (
              <span>✅ Iteration {msg.data?.iteration} complete (score: {msg.data?.score})</span>
            )}
          </div>
        ))}
      </div>
      
      {status === "completed" && (
        <div className="text-green-600 font-medium">
          ✅ Analysis complete! Final score: {currentScore}/10
        </div>
      )}
      
      {status === "failed" && (
        <div className="text-red-600 font-medium">
          ❌ Analysis failed. Please try again.
        </div>
      )}
    </div>
  );
}
```

```typescript
// hooks/useRunPolling.ts
import { useQuery } from "@tanstack/react-query";
import { getRunStatus } from "@/lib/api/runs";
import { RunStatus } from "@/types/runs";

interface UseRunPollingOptions {
  /** Polling interval in milliseconds (default: 15000 = 15 seconds) */
  pollInterval?: number;
  /** Whether to enable polling (default: true) */
  enabled?: boolean;
}

export function useRunPolling(
  runId: string,
  options: UseRunPollingOptions = {}
) {
  const { pollInterval = 15000, enabled = true } = options;

  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRunStatus(runId),
    enabled: enabled && !!runId,
    // Poll while run is in progress
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Stop polling when run is finished
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false;
      }
      return pollInterval;
    },
    // Keep polling even when window is not focused
    refetchIntervalInBackground: true,
    // Don't refetch on window focus - use interval instead
    refetchOnWindowFocus: false,
  });

  const run = query.data;

  return {
    // Query state
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    
    // Run data
    run,
    status: run?.status ?? "pending",
    currentIteration: run?.current_iteration ?? 0,
    maxIterations: run?.max_iterations ?? 5,
    currentPhase: run?.current_phase,
    groundingScores: run?.grounding_scores ?? [],
    errorMessage: run?.error_message,
    
    // Derived state
    isFinished: ["completed", "failed", "cancelled"].includes(run?.status ?? ""),
    isRunning: run?.status === "running",
    latestScore: run?.grounding_scores?.length 
      ? run.grounding_scores[run.grounding_scores.length - 1] 
      : null,
    
    // Actions
    refetch: query.refetch,
  };
}
```

```typescript
// components/runs/RunStatus.tsx
"use client";

import { useRunPolling } from "@/hooks/useRunPolling";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cancelRun } from "@/lib/api/runs";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

interface RunStatusProps {
  runId: string;
}

const PHASE_LABELS: Record<string, string> = {
  pending: "Queued",
  query_analysis: "Analyzing query",
  prompt_generation: "Generating prompts",
  inference: "Running STACK inference",
  evaluation: "Evaluating predictions",
  output_generation: "Generating outputs",
};

export function RunStatus({ runId }: RunStatusProps) {
  const queryClient = useQueryClient();
  
  const {
    status,
    currentIteration,
    maxIterations,
    currentPhase,
    groundingScores,
    errorMessage,
    isLoading,
    isRunning,
    isFinished,
    latestScore,
  } = useRunPolling(runId);

  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner />
        <span className="ml-2">Loading run status...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {isRunning && <LoadingSpinner size="sm" />}
          <span className="text-lg font-medium">
            {status === "completed" && "✅ Completed"}
            {status === "failed" && "❌ Failed"}
            {status === "cancelled" && "⏹️ Cancelled"}
            {status === "running" && PHASE_LABELS[currentPhase ?? ""] ?? "Running"}
            {status === "pending" && "⏳ Queued"}
          </span>
        </div>
        
        {isRunning && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => cancelMutation.mutate()}
            disabled={cancelMutation.isPending}
          >
            {cancelMutation.isPending ? "Cancelling..." : "Cancel Run"}
          </Button>
        )}
      </div>

      {/* Progress Bar */}
      {!isFinished && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-gray-600">
            <span>Iteration {currentIteration} of {maxIterations}</span>
            {latestScore !== null && <span>Score: {latestScore}/10</span>}
          </div>
          <ProgressBar 
            value={currentIteration} 
            max={maxIterations}
            phase={currentPhase}
          />
        </div>
      )}

      {/* Grounding Scores History */}
      {groundingScores.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Grounding Scores</h3>
          <div className="flex space-x-2">
            {groundingScores.map((score, idx) => (
              <div
                key={idx}
                className={`
                  px-3 py-1 rounded-full text-sm font-medium
                  ${score >= 7 ? "bg-green-100 text-green-800" : 
                    score >= 5 ? "bg-yellow-100 text-yellow-800" : 
                    "bg-red-100 text-red-800"}
                `}
              >
                Iter {idx + 1}: {score}/10
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {status === "failed" && errorMessage && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium">Error</h3>
          <p className="text-red-700 text-sm mt-1">{errorMessage}</p>
        </div>
      )}

      {/* Email Notification Note */}
      {isRunning && (
        <p className="text-sm text-gray-500">
          💌 You'll receive an email when this run completes. Feel free to close this page.
        </p>
      )}
    </div>
  );
}
```

---

## 11. Configuration

### 11.1 Backend Configuration (Dynaconf)

```yaml
# backend/settings.yml

default:
  app_name: "HAYSTACK"
  version: "1.0.0"
  debug: false
  log_level: "INFO"
  
  # LLM defaults
  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-5-20250929"
    temperature: 0.7
    max_tokens: 4096
  
  # Iteration defaults
  iteration:
    max_iterations: 5
    score_threshold: 7
    plateau_window: 3
    min_improvement: 1
  
  # Database API settings
  database_apis:
    max_retries: 3
    base_delay_seconds: 1.0
    max_delay_seconds: 30.0
    requests_per_minute: 30
  
  # GCP Batch configuration for STACK inference
  batch:
    region: "us-east1"
    job_timeout_seconds: 1800  # 30 minutes max
    poll_interval_seconds: 10
    machine_type: "a2-highgpu-1g"  # NVIDIA A100 80GB
    accelerator_type: "nvidia-tesla-a100"
    accelerator_count: 1
    boot_disk_size_gb: 200
    container_image: "us-east1-docker.pkg.dev/arc-prod/haystack/stack-inference:latest"
  
  # Email notifications
  email:
    enabled: true
    from_address: "haystack@arc.institute"
    from_name: "HAYSTACK"

dev:
  debug: true
  log_level: "DEBUG"
  
  email:
    enabled: false  # Disable emails in development
  
  database:
    instance_connection_name: "arc-dev:us-east1:haystack-dev"
    database_name: "haystack"
    user: "haystack_app"
  
  gcs:
    project_id: "arc-dev"
    bucket_name: "haystack-dev"
    atlases_prefix: "atlases/"
    stack_model_prefix: "models/stack/"
    results_prefix: "results/"
    batch_io_prefix: "batch-io/"  # For Batch job input/output
  
  batch:
    container_image: "us-east1-docker.pkg.dev/arc-dev/haystack/stack-inference:latest"

prod:
  database:
    instance_connection_name: "arc-prod:us-east1:haystack-prod"
    database_name: "haystack"
    user: "haystack_app"
  
  gcs:
    project_id: "arc-prod"
    bucket_name: "haystack-prod"
    atlases_prefix: "atlases/"
    stack_model_prefix: "models/stack/"
    results_prefix: "results/"
    batch_io_prefix: "batch-io/"
```

### 11.2 Environment Variables

```bash
# .env.example

# Environment selection
DYNACONF=dev  # or prod

# LLM API Keys (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Email notifications (SendGrid)
SENDGRID_API_KEY=SG...

# Database (Cloud SQL)
# Connection handled via Cloud SQL Connector with IAM auth

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=arc-dev

# Frontend
FRONTEND_OUT_DIR=/app/frontend/out
NEXT_PUBLIC_API_URL=https://haystack.arc.institute/api

# Application
PORT=8080
```

---

## 12. Deployment

### 12.1 Docker Configuration (Web Application)

```dockerfile
# Dockerfile (Cloud Run web application)

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN rm -f .env .env.* || true
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend /app/backend
RUN pip install --no-cache-dir -e /app/backend/.

# Copy built frontend
COPY --from=frontend-build /app/frontend/out /app/frontend/out

# Set runtime environment
ENV FRONTEND_OUT_DIR=/app/frontend/out PORT=8080

EXPOSE 8080
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 12.2 Docker Configuration (STACK Inference)

```dockerfile
# Dockerfile.stack (GCP Batch inference container)

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
COPY stack_inference/run_inference.py /app/run_inference.py

WORKDIR /app
ENTRYPOINT ["python", "/app/run_inference.py"]
```

### 12.3 STACK Inference Script

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

## 13. Output Specification

### 13.1 Output Files

| File | Format | Contents |
|------|--------|----------|
| `{run_id}_predictions.h5ad` | AnnData | Predicted expression with metadata |
| `{run_id}_report.html` | HTML | Interactive interpretation report |
| `{run_id}_log.json` | JSON | Complete execution log |

### 13.2 AnnData Structure

```python
predictions.obs:
    - cell_id: str
    - cell_type_query: str
    - perturbation_query: str
    - prompt_strategy: str
    - iteration: int
    - grounding_score: int

predictions.var:
    - gene_symbol: str
    - predicted_lfc: float
    - predicted_pval: float
    - is_de: bool
    - target_gene: bool

predictions.uns:
    - run_id: str
    - query: str
    - config: dict
    - iterations: list[dict]
    - enrichment_results: dict
```

---

## 14. Error Handling

### 14.1 Error Types

```python
class HaystackError(Exception):
    """Base exception for HAYSTACK errors."""
    pass


class QueryParsingError(HaystackError):
    """Failed to parse user query."""
    pass


class CellRetrievalError(HaystackError):
    """Failed to retrieve cells from database."""
    pass


class InferenceError(HaystackError):
    """STACK inference failed."""
    pass


class EvaluationError(HaystackError):
    """Grounding evaluation failed."""
    pass


class ExternalAPIError(HaystackError):
    """External biological database API failed."""
    pass
```

### 14.2 Error Response Format

```python
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    error_type: str
    message: str
    details: Optional[dict] = None
    run_id: Optional[str] = None
    
    
# Example error response
{
    "error": "inference_failed",
    "error_type": "InferenceError",
    "message": "STACK inference timed out after 30 minutes",
    "details": {
        "iteration": 2,
        "prompt_strategy": "mechanistic"
    },
    "run_id": "hay_20260110_abc123"
}
```

---

## 15. Testing Strategy

### 15.1 Backend Tests

```python
# backend/tests/conftest.py

import pytest
import asyncio
from httpx import AsyncClient
from backend.main import app
from backend.services.database import database


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def mock_database(mocker):
    """Mock database for unit tests."""
    mock_db = mocker.MagicMock()
    mocker.patch("backend.services.database.database", mock_db)
    return mock_db
```

```python
# backend/tests/test_api/test_runs.py

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_run(client: AsyncClient, mock_database):
    """Test creating a new run."""
    response = await client.post(
        "/api/v1/runs/",
        json={"query": "How would lung fibroblasts respond to TGF-beta?"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_get_run_status(client: AsyncClient, mock_database):
    """Test getting run status."""
    mock_database.get_run.return_value = {
        "run_id": "test_123",
        "status": "running",
        "iterations": [],
        "config": {"max_iterations": 5},
    }
    
    response = await client.get("/api/v1/runs/test_123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test_123"
    assert data["status"] == "running"
```

### 15.2 Frontend Tests

```typescript
// frontend/__tests__/components/RunForm.test.tsx

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RunForm } from "@/components/runs/RunForm";

const queryClient = new QueryClient();

describe("RunForm", () => {
  it("renders query input", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    expect(screen.getByPlaceholderText(/lung fibroblasts/i)).toBeInTheDocument();
  });
  
  it("disables submit for short queries", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    const button = screen.getByRole("button", { name: /start analysis/i });
    expect(button).toBeDisabled();
  });
});
```

---

## 16. Dependencies

### 16.1 Backend Dependencies

```toml
# backend/pyproject.toml

[project]
name = "haystack"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    # Web framework
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    
    # Validation & settings
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "dynaconf>=3.2.4",
    
    # Database
    "asyncpg>=0.29.0",
    "pgvector>=0.2.0",
    "cloud-sql-python-connector[asyncpg]>=1.6.0",
    
    # Google Cloud
    "google-cloud-storage>=2.14.0",
    "google-cloud-secret-manager>=2.18.0",
    "google-cloud-batch>=0.17.0",
    
    # Email notifications
    "sendgrid>=6.11.0",
    
    # Agent framework
    "langchain>=1.0.0",
    "deepagents>=0.1.0",
    "langgraph>=0.1.0",
    
    # LLM providers
    "langchain-anthropic>=0.1.0",
    "langchain-openai>=0.1.0",
    "langchain-google-genai>=0.1.0",
    
    # Data handling
    "scanpy>=1.10.0",
    "anndata>=0.10.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # API clients
    "httpx>=0.27.0",
    "tenacity>=8.0.0",
    
    # Biological analysis
    "gseapy>=1.0.0",
    "networkx>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]
```

### 16.2 Frontend Dependencies

```json
{
  "name": "haystack-frontend",
  "version": "1.0.0",
  "dependencies": {
    "@tanstack/react-query": "^5.90.10",
    "@headlessui/react": "^2.2.9",
    "@heroicons/react": "^2.2.0",
    "axios": "^1.13.2",
    "clsx": "^2.1.1",
    "date-fns": "^4.1.0",
    "next": "16.0.3",
    "react": "19.2.0",
    "react-dom": "19.2.0",
    "react-markdown": "^9.0.0",
    "tailwind-merge": "^3.4.0",
    "zod": "^4.1.12",
    "zustand": "^4.4.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4",
    "@testing-library/jest-dom": "^6.8.0",
    "@testing-library/react": "^16.3.0",
    "@types/node": "^20",
    "@types/react": "^19",
    "eslint": "^9",
    "eslint-config-next": "16.0.3",
    "jest": "^30.2.0",
    "jest-environment-jsdom": "^30.2.0",
    "tailwindcss": "^4",
    "typescript": "^5"
  }
}
```

---

## 17. Future Extensions

### 17.1 Post-MVP Features

| Feature | Priority | Description |
|---------|----------|-------------|
| User custom atlases | High | Allow users to upload their own H5AD files |
| Results caching | High | Cache API responses and enrichments |
| Batch queries | Medium | Process multiple queries efficiently |
| Validation agent | Medium | Compare predictions to experimental data |
| Jupyter integration | Low | Notebook interface for power users |
| Fine-tuning support | Low | Improve STACK with user feedback |
| **Pause/resume with feedback** | **Medium** | **Allow users to provide mid-run guidance to the agent** |

### 17.2 Pause/Resume with User Feedback (Deferred)

A deferred feature that would allow users to pause a running analysis and provide feedback to the agent:

**Concept:**
- User clicks "Pause for Feedback" during a run
- Agent checkpoints its state and presents current results
- User reviews intermediate predictions and provides guidance (e.g., "Focus more on fibroblast markers", "Exclude immune cell prompts")
- Agent incorporates feedback and resumes

**Why Deferred:**
- Requires agent state checkpointing and resume logic
- Needs UI for structured feedback input
- Timeout handling if user never responds
- Adds significant complexity to the iteration loop

**Alternative (MVP):**
- Users can cancel runs and start new ones with refined queries
- Run history allows comparison across attempts
- Email notification enables asynchronous workflow

### 17.3 Scalability Improvements

- Parallel prompt evaluation across multiple workers
- Distributed vector index with read replicas
- Result streaming with Server-Sent Events (SSE) for active monitoring
- Warm VM pools to reduce Batch job startup latency

### 17.4 Integration Opportunities

- Benchling integration for experimental tracking
- GEO integration for validation data retrieval
- Slack notifications for run completion (alternative to email)
- Asana task creation for failed runs

---

## Appendix A: Grounding Score Calculation

```python
def compute_composite_score(
    pathway: int,
    target: int, 
    literature: int,
    network: int
) -> int:
    """
    Compute composite grounding score.
    
    Args:
        pathway: Pathway coherence score (1-10)
        target: Target activation score (1-10)
        literature: Literature support score (1-10)
        network: Network coherence score (1-10)
    
    Returns:
        Composite score (1-10)
    
    Weights:
    - Pathway coherence: 25%
    - Target activation: 30%
    - Literature support: 25%
    - Network coherence: 20%
    
    Asymmetric penalties:
    - Literature contradictions penalized more heavily than lack of support
    - Novel predictions (no literature) not penalized
    """
    base_score = (
        0.25 * pathway +
        0.30 * target +
        0.25 * literature +
        0.20 * network
    )
    return max(1, min(10, round(base_score)))
```

---

## Appendix B: Retrieval Strategy Hierarchy

See `prompt-retrieval.md` for detailed specification of the cell retrieval strategies:
- Direct Match Strategy
- Mechanistic Match Strategy
- Semantic Match Strategy
- Ontology-Guided Strategy