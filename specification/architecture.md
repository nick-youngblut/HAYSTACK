# Architecture Overview

### 3.1 High-Level Architecture

HAYSTACK uses a **two-tier GCP Batch architecture** for robust, long-running agentic workflows:

1. **Cloud Run** — Thin API layer for job submission, status queries, and frontend serving
2. **CPU Batch Job** — Orchestrator agent running the iterative optimization loop
3. **GPU Batch Job** — STACK inference only (submitted by the orchestrator)

This design ensures runs survive browser disconnects, Cloud Run scale-downs, and enables parallel runs per user.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Google Cloud Run (Thin API Layer)                            │
│                         us-east1                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Backend (uvicorn on port 8080)                │   │
│  │                                                                          │   │
│  │   POST /api/v1/runs/        → Submit CPU Batch job, return run_id        │   │
│  │   GET  /api/v1/runs/{id}    → Read status from Cloud SQL                 │   │
│  │   POST /api/v1/runs/{id}/cancel → Cancel Batch job via API               │   │
│  │   GET  /api/v1/runs/{id}/result → Return signed GCS URLs                 │   │
│  │   (no /api/v1/cells browse) → Cell retrieval happens inside orchestrator │   │
│  │   GET  /api/v1/metadata/*   → Lookup tables                              │   │
│  │   /*                        → Static files (Next.js)                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                 Next.js Frontend (Static Export)                         │   │
│  │   - TanStack Query (polling for status updates)                          │   │
│  │   - Zustand (state management)                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Submits Batch job
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GCP Batch — CPU Orchestrator Job                             │
│                    e2-standard-4 (~$0.13/hr)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                 Orchestrator Agent Container                             │   │
│  │                                                                          │   │
│  │   • Query understanding (LLM calls)                                      │   │
│  │   • Prompt generation (LLM + DB queries)                                 │   │
│  │   • Grounding evaluation (external APIs + LLM)                           │   │
│  │   • Iteration control and convergence checking                           │   │
│  │   • Updates status in Cloud SQL (for polling)                            │   │
│  │   • Submits GPU Batch jobs for STACK inference                           │   │
│  │   • Writes final results to GCS                                          │   │
│  │   • Sends email notification on completion                               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Submits GPU job per iteration
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GCP Batch — GPU Inference Job                                │
│                    a2-highgpu-1g / NVIDIA A100 80GB (~$3/hr)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                 STACK Inference Container                                │   │
│  │                                                                          │   │
│  │   • Load STACK model from GCS                                            │   │
│  │   • Load prompt/query cells from GCS                                     │   │
│  │   • Run forward pass (T=5 diffusion steps)                               │   │
│  │   • Write predictions.h5ad to GCS                                        │   │
│  │   • Exit with status code                                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  GCP Cloud SQL   │         │  GCS Bucket      │         │  External APIs   │
│  (PostgreSQL +   │         │                  │         │                  │
│   pgvector)      │         │  • Atlas H5ADs   │         │  • LLM Providers │
│                  │         │  • STACK model   │         │  • Reactome/KEGG │
│  • Cell metadata │         │  • Batch I/O     │         │  • UniProt       │
│  • Embeddings    │         │  • Results       │         │  • PubMed        │
│  • Run history   │         │                  │         │  • Semantic Scholar │
│                  │         │                  │         │  • bioRxiv/medRxiv │
│                  │         │                  │         │  • CORE API      │
│                  │         │                  │         │  • Europe PMC    │
│                  │         │                  │         │  • Unpaywall     │
│                  │         │                  │         │  • SendGrid      │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

### 3.2 Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HAYSTACK REQUEST FLOW                                 │
│                     (Two-Tier GCP Batch Architecture)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 1. USER SUBMITS QUERY (Cloud Run API)                                   │    │
│  │    • POST /api/v1/runs/ with natural language query                     │    │
│  │    • User email extracted from IAP headers                              │    │
│  │    • Cloud Run creates run record in Cloud SQL (status: "pending")      │    │
│  │    • Cloud Run submits CPU Batch job for orchestrator                   │    │
│  │    • Returns run_id immediately; frontend polls for status              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       │  Cloud Run submits GCP Batch job                                        │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 2. CPU BATCH JOB: ORCHESTRATOR AGENT (e2-standard-4)                    │    │
│  │    Runs entire agentic workflow in isolated container                   │    │
│  │                                                                          │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2a. QUERY UNDERSTANDING                                         │   │    │
│  │    │     • Parse query → StructuredQuery                             │   │    │
│  │    │     • Resolve cell type (CL ontology)                           │   │    │
│  │    │     • Resolve perturbation (DrugBank, PubChem)                  │   │    │
│  │    │     • Retrieve biological priors                                │   │    │
│  │    │     • Update Cloud SQL: phase = "query_analysis"                │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                          │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2b. PROMPT GENERATION (per iteration)                           │   │    │
│  │    │     • Run parallel retrieval strategies                         │   │    │
│  │    │     • Rank and select prompt candidates                         │   │    │
│  │    │     • Update Cloud SQL: phase = "prompt_generation"             │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                                │    │
│  │         │  Orchestrator submits GPU Batch job                            │    │
│  │         ▼                                                                │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2c. STACK INFERENCE (GPU Batch Job - A100 80GB)                 │   │    │
│  │    │     • Write prompt/query cells to GCS                           │   │    │
│  │    │     • Submit GPU Batch job                                      │   │    │
│  │    │     • Poll for completion (10s intervals)                       │   │    │
│  │    │     • Read predictions from GCS                                 │   │    │
│  │    │     • Update Cloud SQL: phase = "inference"                     │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                                │    │
│  │         ▼                                                                │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2d. GROUNDING EVALUATION                                        │   │    │
│  │    │     • Extract DE genes from predictions                         │   │    │
│  │    │     • Run pathway enrichment (GO, KEGG, Reactome)               │   │    │
│  │    │     • Check literature support via PubMed/Semantic Scholar      │   │    │
│  │    │     • Compute grounding score (1-10)                            │   │    │
│  │    │     • Update Cloud SQL: phase = "evaluation", scores            │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                                │    │
│  │         ▼                                                                │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2e. ITERATION CONTROL                                           │   │    │
│  │    │     • Check Cloud SQL for cancellation flag                     │   │    │
│  │    │     • Check convergence: score ≥ threshold OR max iterations    │   │    │
│  │    │     • If not converged: loop back to 2b                         │   │    │
│  │    │     • If converged: proceed to output generation                │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                                │    │
│  │         ▼                                                                │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │    │ 2f. OUTPUT GENERATION & NOTIFICATION                            │   │    │
│  │    │     • Write AnnData with predictions to GCS                     │   │    │
│  │    │     • Generate interpretation report                            │   │    │
│  │    │     • Update Cloud SQL: status = "completed"                    │   │    │
│  │    │     • Send email notification via SendGrid                      │   │    │
│  │    │     • Exit with success code                                    │   │    │
│  │    └─────────────────────────────────────────────────────────────────┘   │    │
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
│  POST /api/v1/runs/ ──────► Cloud Run submits CPU Batch job                     │
│                             Returns { run_id, status: "pending" }               │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Poll every 15 seconds:                                                 │    │
│  │  GET /api/v1/runs/{run_id}                                              │    │
│  │                                                                         │    │
│  │  Cloud Run reads from Cloud SQL (updated by CPU Batch job):             │    │
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
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  CANCELLATION FLOW:                                                             │
│  POST /api/v1/runs/{run_id}/cancel                                              │
│       │                                                                         │
│       ▼                                                                         │
│  Cloud Run sets cancellation flag in Cloud SQL                                  │
│  Cloud Run calls Batch API to delete CPU orchestrator job                       │
│  (GPU job, if running, is also cancelled)                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Component Responsibilities

| Component | Responsibility | Technology | Runs In |
|-----------|---------------|------------|---------|
| Frontend | User interface, query input, status polling, result visualization | Next.js, TypeScript, TanStack Query | Cloud Run |
| API Layer | Job submission, status queries, cancellation, static file serving | FastAPI, Pydantic | Cloud Run |
| Orchestrator Agent | Full agentic workflow: query analysis, prompt generation, iteration control | LangChain, DeepAgents | CPU Batch Job |
| Query Understanding | Parses queries, resolves entities, retrieves priors | LangChain tools | CPU Batch Job |
| Prompt Generation | Generates and ranks prompt candidates | LangChain tools | CPU Batch Job |
| Grounding Evaluation | Evaluates predictions, computes scores | LangChain tools | CPU Batch Job |
| STACK Inference | GPU-accelerated model inference | PyTorch, STACK | GPU Batch Job |
| Database | Cell metadata, vector embeddings, run history/status | Cloud SQL (PostgreSQL + pgvector) | GCP Managed |
| Object Storage | Atlas H5AD files, STACK model, Batch I/O, results | GCS | GCP Managed |
| Email Service | Completion/failure notifications | SendGrid | CPU Batch Job |

### 3.4 Two-Tier Batch Job Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TWO-TIER BATCH JOB ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Cloud Run (API)              CPU Batch (Orchestrator)    GPU Batch (STACK)     │
│  ───────────────              ────────────────────────    ─────────────────     │
│                                                                                 │
│  1. Receive POST /runs/                                                         │
│     • Validate request                                                          │
│     • Create run in DB                                                          │
│            │                                                                    │
│            ▼                                                                    │
│  2. Submit CPU Batch job ─────────► Job starts                                  │
│     • Pass run_id, query           • Connect to Cloud SQL                       │
│     • Return run_id to user        • Update status: "running"                   │
│            │                       • Begin agent workflow                       │
│            │                              │                                     │
│  3. Poll GET /runs/{id}                   │                                     │
│     • Read status from DB                 │                                     │
│     • Return to frontend                  │                                     │
│            │                              │                                     │
│            │                              ▼                                     │
│            │               [Iteration N]                                        │
│            │                 • Generate prompts                                 │
│            │                 • Write cells to GCS                               │
│            │                        │                                           │
│            │                        ▼                                           │
│            │               Submit GPU job ──────────► Job starts                │
│            │                 • Poll for completion    • Load STACK model        │
│            │                 • 10s intervals          • Run inference           │
│            │                        │                 • Write to GCS            │
│            │                        │                        │                  │
│            │                        ◄────────────────────────┘                  │
│            │                 • Read predictions                                 │
│            │                 • Evaluate grounding                               │
│            │                 • Update DB with scores                            │
│            │                 • Check convergence                                │
│            │                        │                                           │
│            │                        ▼                                           │
│            │               [If not converged: loop]                             │
│            │               [If converged: continue]                             │
│            │                        │                                           │
│            │                        ▼                                           │
│            │               Output generation                                    │
│            │                 • Write results to GCS                             │
│            │                 • Update DB: "completed"                           │
│            │                 • Send email                                       │
│            │                 • Exit                                             │
│            │                                                                    │
│  4. GET /runs/{id}/result                                                       │
│     • Generate signed URLs                                                      │
│     • Return to user                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

COST EFFICIENCY:
┌────────────────────┬─────────────────┬──────────────────────────────────────────┐
│ Component          │ Hourly Cost     │ Usage Pattern                            │
├────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ CPU Orchestrator   │ ~$0.13/hr       │ Runs for entire workflow (10-30 min)     │
│ (e2-standard-4)    │                 │ Active during LLM calls, DB queries      │
├────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ GPU Inference      │ ~$3.00/hr       │ Only during STACK inference (~2-5 min    │
│ (A100 80GB)        │                 │ per iteration, 1-5 iterations)           │
├────────────────────┼─────────────────┼──────────────────────────────────────────┤
│ Cloud Run          │ ~$0.00024/req   │ Minimal: job submission + status polls   │
└────────────────────┴─────────────────┴──────────────────────────────────────────┘
```

### 3.5 State Management

HAYSTACK uses a combination of state management approaches:

- **Database state**: Run metadata, status, iteration progress stored in Cloud SQL (shared between Cloud Run and Batch jobs)
- **Object storage state**: Atlas files, STACK checkpoints, Batch I/O, and result artifacts stored in GCS
- **Batch job communication**: CPU orchestrator → GPU inference via GCS (prompt/query cells in, predictions out)
- **In-memory state**: Agent execution state maintained via LangGraph checkpointing within the CPU Batch job
- **Frontend state**: Run status and UI state managed via Zustand stores
- **Cancellation**: Cloud Run sets flag in Cloud SQL; CPU Batch job checks flag between iterations

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
- `specification/frontend.md`
- `specification/database.md`
- `specification/deployment.md`
