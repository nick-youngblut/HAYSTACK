# HAYSTACK: Iterative Knowledge-Guided Cell Prompting System

*Finding the optimal prompt in a haystack of possibilities*

## Overview

HAYSTACK is a closed-loop agentic system that iteratively generates and evaluates cell prompts for STACK inference. It couples multi-strategy prompt generation with biological grounding evaluation (pathway knowledge, literature, and priors), feeding scores back into the next iteration until convergence or a stopping criterion is reached.

At a high level, the system runs as a FastAPI + Next.js application on Cloud Run, with long-running agent workflows executed in GCP Batch (CPU orchestrator + GPU inference). A shared Cloud SQL database tracks run state while GCS stores models, atlases, and outputs.

## MVP Goals (High-Level)

- Accept natural language ICL requests via a web interface
- Generate biologically informed prompts using multiple retrieval strategies
- Run STACK inference and evaluate predictions against biological knowledge
- Iterate until convergence with configurable thresholds
- Provide polling status, cancellation, and downloadable results

## Specification Map

- Architecture: `specification/architecture.md`
- Data models: `specification/data-models.md`
- Agent specs: `specification/agents.md`
- Tooling specs: `specification/tools.md`
- Biological database integration: `specification/biological-database-integration.md`
- Database schema and storage: `specification/database.md`
- Backend API: `specification/backend-api.md`
- Orchestrator (CPU Batch): `specification/orchestrator.md`
- Frontend: `specification/frontend.md`
- Configuration: `specification/configuration.md`
- Deployment: `specification/deployment.md`
- Output artifacts: `specification/output.md`
- Error handling: `specification/error-handling.md`
- Testing: `specification/testing.md`
- Dependencies: `specification/dependencies.md`
- Future extensions: `specification/future-extensions.md`
- Literature search module: `specification/literature-search.md`
- Prompt retrieval strategies: `specification/prompt-retrieval.md`
- Appendices: `specification/appendices.md`
