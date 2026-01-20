# HAYSTACK Implementation Plan Overview

**Project**: HAYSTACK - Heuristic Agent for Yielding STACK-Tuned Assessments with Closed-loop Knowledge  
**Duration**: 14-18 weeks total  
**Last Updated**: 2026-01-19

---

## Executive Summary

This implementation plan breaks down the HAYSTACK system into 12 sprints, each focused on a specific component or capability. The plan is derived from `INIT-IMP-PLAN.md` and expands each phase with detailed tasks, code examples, and references to the specification documents in `./specification/`.

---

## Sprint Overview

| Sprint | Name | Duration | Key Deliverables |
|--------|------|----------|------------------|
| 01 | Foundation & Infrastructure | 2-3 weeks | GCP setup, Cloud SQL + pgvector, Atlas data loading |
| 02 | Core Backend Services | 1-2 weeks | Pydantic models, database client, GCS service, APIs |
| 03 | Prompt Retrieval Strategies | 2 weeks | 6 retrieval strategies, ranking, control matching |
| 04 | Agent Framework | 2-3 weeks | LangChain tools, 3 subagents, orchestrator agent |
| 05 | STACK Inference Integration | 1 week | GPU Batch job, inference container, batch client |
| 06 | Cloud Run API | 1 week | FastAPI endpoints, CPU batch client, IAP auth |
| 07 | Frontend | 2 weeks | Next.js app, run management, status polling |
| 08 | Orchestrator Batch Job | 1-2 weeks | Entrypoint, email service, output generation |
| 09 | Docker Containers | 1 week | API, orchestrator, inference Dockerfiles |
| 10 | Deployment & Configuration | 1 week | Cloud Run, Dynaconf, Secret Manager, monitoring |
| 11 | Testing | 2 weeks | Unit, integration, load tests |
| 12 | Documentation & Launch | 1 week | API docs, user guide, security review |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER (Browser)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CLOUD RUN (Sprint 06/07/09)                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FastAPI + Next.js Static Export                            │    │
│  │  • POST /api/v1/runs/ → Submit CPU Batch job                │    │
│  │  • GET /api/v1/runs/{id} → Poll status from Cloud SQL       │    │
│  │  • GET /api/v1/runs/{id}/result → Generate signed URLs      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  CPU BATCH (Sprint 04/08)      │   │  CLOUD SQL (Sprint 01/02)     │
│  ┌───────────────────────────┐ │   │  ┌───────────────────────────┐│
│  │ OrchestratorAgent         │ │   │  │ PostgreSQL 15 + pgvector  ││
│  │ • QueryUnderstandingAgent │ │◄──┼──│ • cells (~10M rows)       ││
│  │ • PromptGenerationAgent   │ │   │  │ • ontology_terms (~2.5K)  ││
│  │ • GroundingEvaluationAgent│ │   │  │ • runs (history)          ││
│  └───────────────────────────┘ │   │  └───────────────────────────┘│
└───────────────────────────────┘   └───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  GPU BATCH (Sprint 05/09)     │   │  GCS (Sprint 01)              │
│  ┌───────────────────────────┐│   │  ┌───────────────────────────┐│
│  │ STACK Model Inference     ││◄──┼──│ • haystack-atlases        ││
│  │ • A100 80GB GPU           ││   │  │ • haystack-models         ││
│  │ • Diffusion steps         ││───┼──│ • haystack-batch-io       ││
│  └───────────────────────────┘│   │  │ • haystack-results        ││
└───────────────────────────────┘   │  └───────────────────────────┘│
                                    └───────────────────────────────┘
```

---

## Key Configuration Values

Per `./specification/` documents:

| Parameter | Value | Source |
|-----------|-------|--------|
| **Task Types** | 5 ICL types | `data-models.md` |
| **Control Strategies** | synthetic_control, query_as_control | `data-models.md` |
| **Ranking Weights** | Relevance: 0.4, Quality: 0.3, Diversity: 0.3 | `prompt-retrieval.md` |
| **Embedding Model** | text-embedding-3-small (1536 dim) | `prompt-retrieval.md` |
| **Max Iterations** | 5 | `configuration.md` |
| **Score Threshold** | 7/10 | `configuration.md` |
| **Default LLM** | claude-sonnet-4-5-20250929 | `configuration.md` |

---

## Critical Path

```
Sprint 01 ──┬── Sprint 02 ──┬── Sprint 03 ── Sprint 04 ──┬── Sprint 05
            │               │                            │
            │               └── Sprint 06 ───────────────┤
            │                                            │
            └────────────────────────────────────────────┼── Sprint 08
                                                         │
Sprint 07 (parallel with 05-06) ─────────────────────────┤
                                                         │
                                              Sprint 09 ─┴── Sprint 10
                                                              │
                                                    Sprint 11 ── Sprint 12
```

---

## Documentation Index

### Sprint Documents

1. [01_foundation_infrastructure.md](./01_foundation_infrastructure.md) - GCP, Database, Atlas Data
2. [02_core_backend_services.md](./02_core_backend_services.md) - Models, Clients, Services
3. [03_prompt_retrieval_strategies.md](./03_prompt_retrieval_strategies.md) - Retrieval Strategies
4. [04_agent_framework.md](./04_agent_framework.md) - LangChain Agents & Tools
5. [05_stack_inference_integration.md](./05_stack_inference_integration.md) - GPU Batch Inference
6. [06_cloud_run_api.md](./06_cloud_run_api.md) - FastAPI Backend
7. [07_frontend.md](./07_frontend.md) - Next.js Frontend
8. [08_orchestrator_batch_job.md](./08_orchestrator_batch_job.md) - CPU Batch Workflow
9. [09_docker_containers.md](./09_docker_containers.md) - Docker Configuration
10. [10_deployment_configuration.md](./10_deployment_configuration.md) - GCP Deployment
11. [11_testing.md](./11_testing.md) - Test Suite
12. [12_documentation_launch_prep.md](./12_documentation_launch_prep.md) - Launch Prep

### Specification Documents

Located in `./specification/`:

| Document | Purpose |
|----------|---------|
| `overview.md` | Project overview and motivation |
| `architecture.md` | System architecture diagrams |
| `agents.md` | LangChain agent design |
| `database.md` | Cloud SQL schema and queries |
| `prompt-retrieval.md` | Retrieval strategies and ranking |
| `orchestrator.md` | CPU batch workflow |
| `ontology-resolution.md` | Cell Ontology integration |
| `tools.md` | LangChain tool definitions |
| `frontend.md` | Next.js components |
| `backend-api.md` | FastAPI endpoints |
| `deployment.md` | Docker and GCP deployment |
| `testing.md` | Test strategy |
| `literature-search.md` | Literature search APIs |
| `data-models.md` | Pydantic models |
| `configuration.md` | Dynaconf settings |
| `dependencies.md` | Python and npm packages |

---

## Technology Stack

### Backend
- **Python 3.11+**
- **FastAPI** - API framework
- **LangChain + DeepAgents** - Agent framework
- **asyncpg + pgvector** - Database client
- **Pydantic v2** - Data validation
- **Dynaconf** - Configuration

### Frontend
- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **TanStack Query** - Data fetching
- **Tailwind CSS v4** - Styling
- **Zustand** - State management

### Infrastructure
- **GCP Cloud Run** - API hosting
- **GCP Batch** - CPU/GPU jobs
- **GCP Cloud SQL** - PostgreSQL 15 + pgvector
- **GCP Cloud Storage** - Artifacts
- **GCP Secret Manager** - Secrets

### AI/ML
- **STACK** - Cell expression prediction model
- **OpenAI text-embedding-3-small** - Embeddings
- **Claude/GPT-4** - Agent LLMs

---

## Estimated Costs (Monthly)

| Resource | Estimate |
|----------|----------|
| Cloud SQL (db-custom-4-15360) | ~$300 |
| Cloud Run (min 0, max 10) | ~$50 |
| GPU Batch (100 runs × 10 min) | ~$150 |
| Cloud Storage (500 GB) | ~$15 |
| LLM API (100 runs × 5 iter) | ~$50 |
| **Total** | **~$565/month** |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Atlas download failures | High | Multiple sources, local cache |
| LLM API rate limits | Medium | Retry logic, fallback providers |
| STACK inference OOM | Medium | Tune batch size, monitor GPU memory |
| Cell type resolution gaps | Medium | OLS fallback, log unmapped types |
| VPC configuration issues | High | Follow GCP docs, test early |

---

## Success Criteria

1. **Functional**: Complete workflow executes for all 5 task types
2. **Performance**: <2 min for query understanding, <5 min for inference
3. **Quality**: Grounding scores average ≥6/10 on test queries
4. **Reliability**: <5% run failure rate
5. **Security**: IAP authentication working, no exposed secrets
