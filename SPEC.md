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
8. [Vector Database Specification](#8-vector-database-specification)
9. [SLURM Integration](#9-slurm-integration)
10. [CLI Interface](#10-cli-interface)
11. [Configuration](#11-configuration)
12. [Output Specification](#12-output-specification)
13. [Error Handling](#13-error-handling)
14. [Testing Strategy](#14-testing-strategy)
15. [Dependencies](#15-dependencies)
16. [Future Extensions](#16-future-extensions)

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

- Natural language query interface for perturbation prediction requests
- Multi-strategy prompt generation leveraging drug-target knowledge, cell ontologies, and vector similarity
- Automated biological grounding evaluation with integer scoring (1-10)
- Iterative refinement with configurable stopping criteria
- SLURM cluster integration for GPU-accelerated STACK inference
- Support for Claude, OpenAI (GPT-5.2), and Gemini language models

---

## 2. System Goals and Non-Goals

### 2.1 Goals (MVP)

| Goal | Description |
|------|-------------|
| G1 | Accept natural language queries describing perturbation prediction tasks |
| G2 | Generate biologically-informed prompts using multiple parallel strategies |
| G3 | Execute STACK inference on SLURM cluster with H100 GPUs |
| G4 | Evaluate predictions against biological knowledge bases |
| G5 | Iteratively refine prompts based on grounding evaluation |
| G6 | Produce interpretable outputs (AnnData, reports, logs) |
| G7 | Support multiple LLM backends (Claude, OpenAI, Gemini) |

### 2.2 Non-Goals (MVP)

| Non-Goal | Rationale |
|----------|-----------|
| NG1 | User-provided custom datasets in vector index | Simplifies MVP; use fixed atlases |
| NG2 | Real-time interactive mode | CLI batch mode is sufficient for MVP |
| NG3 | Fine-tuning STACK model | Out of scope; use pre-trained checkpoints |
| NG4 | Multi-species support | STACK is human-only currently |
| NG5 | Computational efficiency optimizations | Correctness first; optimize later |
| NG6 | Results caching (SQLite cache) | MVP stores state in memory and filesystem only |

### 2.3 Success Criteria

1. System produces biologically grounded predictions for >70% of queries
2. Iterative refinement improves grounding scores in >50% of cases where initial score < 7
3. End-to-end latency < 2 hours for typical queries on SLURM cluster
4. All outputs are reproducible given the same random seed and configuration

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HAYSTACK System                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATOR AGENT                               │   │
│  │                    (DeepAgent on CPU/Login Node)                         │   │
│  │                                                                          │   │
│  │  • Manages iteration loop                                                │   │
│  │  • Coordinates subagents                                                 │   │
│  │  • Handles convergence checking                                          │   │
│  │  • Produces final outputs                                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                           │
│           ▼                    ▼                    ▼                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                     │
│  │    QUERY       │  │    PROMPT      │  │   GROUNDING    │                     │
│  │ UNDERSTANDING  │  │  GENERATION    │  │  EVALUATION    │                     │
│  │   SUBAGENT     │  │   SUBAGENT     │  │   SUBAGENT     │                     │
│  └────────────────┘  └────────────────┘  └────────────────┘                     │
│           │                    │                    │                           │
│           ▼                    ▼                    ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           TOOL LAYER                                     │   │
│  │                                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Drug-Target │  │    Cell     │  │   Vector    │  │  Pathway    │      │   │
│  │  │  Knowledge  │  │  Ontology   │  │  Database   │  │  Analysis   │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  │                                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Literature  │  │   STRING    │  │    STACK    │  │   SLURM     │      │   │
│  │  │   Search    │  │     PPI     │  │  Inference  │  │   Submit    │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA LAYER                                       │   │
│  │                                                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │   │
│  │  │  Vector Index   │  │  Atlas Storage  │  │ Iteration State │           │   │
│  │  │  (LanceDB)      │  │    (H5AD)       │  │  (Filesystem)   │           │   │
│  │  │                 │  │                 │  │                 │           │   │
│  │  │ • Parse PBMC    │  │ • Parse PBMC    │  │ • Per-iteration │           │   │
│  │  │ • OpenProblems  │  │ • OpenProblems  │  │   AnnData       │           │   │
│  │  │ • Tabula Sapiens│  │ • Tabula Sapiens│  │ • Evaluation    │           │   │
│  │  │                 │  │                 │  │   results       │           │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SLURM CLUSTER                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    GPU PARTITION (H100 80GB)                            │    │
│  │                                                                         │    │
│  │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │    │
│  │    │ STACK Job 1  │    │ STACK Job 2  │    │ STACK Job N  │             │    │
│  │    │ (Iteration 1)│    │ (Iteration 2)│    │ (Iteration N)│             │    │
│  │    └──────────────┘    └──────────────┘    └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Query (Natural Language)                                                  │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 1. QUERY UNDERSTANDING                                                  │    │
│  │    • Parse intent (perturbation type, cell type, condition)             │    │
│  │    • Resolve entities (map to ontology IDs, gene symbols)               │    │
│  │    • Retrieve prior knowledge (targets, expected pathways)              │    │
│  │    • Assess feasibility                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 2. PROMPT GENERATION (Parallel Strategies)                              │    │
│  │                                                                         │    │
│  │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │    │
│  │    │  Direct     │  │ Mechanistic │  │  Semantic   │  │  Ontology   │   │    │
│  │    │  Match      │  │   Match     │  │   (Vector)  │  │  Guided     │   │    │
│  │    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │    │
│  │           │                │                │                │          │    │
│  │           └────────────────┴────────────────┴────────────────┘          │    │
│  │                                    │                                    │    │
│  │                                    ▼                                    │    │
│  │                        Candidate Ranking & Selection                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 3. STACK INFERENCE (SLURM GPU Job)                                      │    │
│  │    • Submit job via submitit                                            │    │
│  │    • Run STACK (Large, T=5) with selected prompts                       │    │
│  │    • Return predicted expression profiles                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 4. BIOLOGICAL GROUNDING EVALUATION                                      │    │
│  │                                                                         │    │
│  │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │    │
│  │    │  Pathway    │  │  Target     │  │ Literature  │  │  Network    │   │    │
│  │    │ Coherence   │  │ Activation  │  │  Support    │  │ Coherence   │   │    │
│  │    │  (1-10)     │  │   (1-10)    │  │   (1-10)    │  │   (1-10)    │   │    │
│  │    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │    │
│  │           │                │                │                │          │    │
│  │           └────────────────┴────────────────┴────────────────┘          │    │
│  │                                    │                                    │    │
│  │                                    ▼                                    │    │
│  │                    Composite Grounding Score (1-10)                     │    │
│  │                    + Detailed Diagnostics                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 5. CONVERGENCE CHECK                                                    │    │
│  │                                                                         │    │
│  │    Score >= threshold (default: 7)?  ──Yes──→  TERMINATE (Success)      │    │
│  │              │                                                          │    │
│  │              No                                                         │    │
│  │              │                                                          │    │
│  │    Iterations >= max (default: 5)?   ──Yes──→  TERMINATE (Max Reached)  │    │
│  │              │                                                          │    │
│  │              No                                                         │    │
│  │              │                                                          │    │
│  │    No improvement over 3 iterations? ──Yes──→  TERMINATE (Plateau)      │    │
│  │              │                                                          │    │
│  │              No                                                         │    │
│  │              │                                                          │    │
│  │              ▼                                                          │    │
│  │    REFINEMENT REASONING                                                 │    │
│  │    • Diagnose failure modes                                             │    │
│  │    • Generate refinement suggestions                                    │    │
│  │    • Update prompt generation context                                   │    │
│  │              │                                                          │    │
│  │              └─────────────────→  Loop to Step 2                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 6. OUTPUT GENERATION                                                    │    │
│  │    • AnnData with predictions and metadata                              │    │
│  │    • Interpretation report (Markdown/HTML)                              │    │
│  │    • JSON execution log                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Component Responsibilities

| Component | Responsibility | Execution Location |
|-----------|---------------|-------------------|
| Orchestrator Agent | Manages iteration loop, coordinates subagents, handles I/O | CPU node / Login node |
| Query Understanding Subagent | Parses queries, resolves entities, retrieves priors | Within orchestrator |
| Prompt Generation Subagent | Generates and ranks prompt candidates | Within orchestrator |
| Grounding Evaluation Subagent | Evaluates predictions, computes scores | Within orchestrator |
| STACK Inference | Runs STACK model forward pass | SLURM GPU node (H100) |
| Vector Database | Stores and retrieves cell embeddings | Shared filesystem |
| Biological Database APIs | Provides pathway, ontology, literature data | External APIs |

### 3.4 State Management

For the MVP, HAYSTACK uses a simple state management approach:

- **In-memory state**: Agent state during execution is maintained in memory via LangGraph's `MemorySaver` checkpointer
- **Filesystem state**: Intermediate results (per-iteration AnnData, evaluation JSONs, SLURM logs) are written to the output directory
- **No caching**: API responses from biological databases are not cached; each run makes fresh API calls

This approach prioritizes simplicity and reproducibility over performance. Future versions may add caching as an optimization.

---

## 4. Data Models

### 4.1 Core Data Models (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class PerturbationType(str, Enum):
    """Types of perturbations supported."""
    DRUG = "drug"
    CYTOKINE = "cytokine"
    GENETIC = "genetic"
    UNKNOWN = "unknown"


class StructuredQuery(BaseModel):
    """Parsed representation of a user query."""
    
    # Core identifiers
    query_id: str = Field(description="Unique identifier for this query")
    raw_query: str = Field(description="Original natural language query")
    
    # Intent
    intent: Literal["perturbation_prediction", "cell_state_generation", "unknown"]
    
    # Cell type information
    query_cell_type: str = Field(description="Target cell type name")
    query_cell_type_cl_id: Optional[str] = Field(
        default=None, 
        description="Cell Ontology ID (e.g., CL:0000235)"
    )
    query_tissue: Optional[str] = Field(default=None, description="Tissue of origin")
    
    # Perturbation information
    perturbation: str = Field(description="Perturbation name")
    perturbation_type: PerturbationType
    perturbation_id: Optional[str] = Field(
        default=None,
        description="External ID (ChEBI, DrugBank, etc.)"
    )
    
    # Biological priors (populated during query understanding)
    target_genes: list[str] = Field(
        default_factory=list,
        description="Known/expected target genes of perturbation"
    )
    expected_pathways: list[str] = Field(
        default_factory=list,
        description="Expected affected pathways"
    )
    
    # Feasibility assessment
    feasibility_score: int = Field(
        default=5,
        ge=1, le=10,
        description="Estimated feasibility of prediction (1-10)"
    )
    feasibility_notes: list[str] = Field(
        default_factory=list,
        description="Notes on feasibility assessment"
    )


class PromptCandidate(BaseModel):
    """A candidate prompt for STACK inference."""
    
    candidate_id: str = Field(description="Unique identifier")
    
    # Source information
    dataset: Literal["parse_pbmc", "openproblems", "tabula_sapiens"]
    perturbation: str
    cell_types: list[str]
    donor_id: Optional[str] = None
    
    # Matching strategy
    match_strategy: Literal["direct", "mechanistic", "semantic", "ontology"]
    
    # Scoring
    relevance_score: int = Field(ge=1, le=10, description="Relevance to query (1-10)")
    
    # Rationale
    rationale: str = Field(description="Why this prompt was selected")
    shared_pathways: list[str] = Field(
        default_factory=list,
        description="Pathways shared with query perturbation"
    )
    shared_targets: list[str] = Field(
        default_factory=list,
        description="Target genes shared with query perturbation"
    )
    
    # Cell indices for retrieval
    prompt_cell_indices: list[int] = Field(description="Indices into atlas file")
    context_cell_indices: list[int] = Field(description="Context cell indices")


class PromptGenerationResult(BaseModel):
    """Result of prompt generation phase."""
    
    iteration: int
    candidates: list[PromptCandidate]
    selected_candidate: PromptCandidate
    selection_rationale: str
    strategies_used: list[str]
    
    # Refinement context (if not first iteration)
    refinement_applied: Optional[str] = None


class DEGene(BaseModel):
    """A differentially expressed gene from predictions."""
    
    gene_symbol: str
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float
    direction: Literal["up", "down"]
    
    # Grounding annotations (populated during evaluation)
    in_expected_targets: bool = False
    pathway_annotations: list[str] = Field(default_factory=list)
    literature_support: Optional[Literal["supported", "contradicted", "novel"]] = None
    literature_citations: list[str] = Field(default_factory=list)


class ComponentScore(BaseModel):
    """Score for a single evaluation component."""
    
    component: str
    score: int = Field(ge=1, le=10)
    rationale: str
    details: dict = Field(default_factory=dict)


class GroundingEvaluation(BaseModel):
    """Complete biological grounding evaluation."""
    
    iteration: int
    
    # Component scores (1-10 integers)
    pathway_coherence: ComponentScore
    target_activation: ComponentScore
    literature_support: ComponentScore
    network_coherence: ComponentScore
    
    # Composite score
    composite_score: int = Field(ge=1, le=10)
    
    # DE gene analysis
    de_genes: list[DEGene]
    num_de_genes: int
    
    # Novel predictions (potentially interesting, not failures)
    novel_predictions: list[str] = Field(
        default_factory=list,
        description="Genes with predictions but no literature support"
    )
    
    # Diagnostics for refinement
    diagnostics: list[str] = Field(
        default_factory=list,
        description="Issues identified that could guide refinement"
    )
    
    # Flags
    is_converged: bool = False
    termination_reason: Optional[str] = None


class RefinementSuggestion(BaseModel):
    """Suggestion for how to refine prompts in next iteration."""
    
    diagnosis: str = Field(description="What went wrong")
    suggested_action: str = Field(description="What to do differently")
    strategy_adjustment: str = Field(description="How to modify prompt generation")
    confidence: int = Field(ge=1, le=10, description="Confidence this will help")


class IterationRecord(BaseModel):
    """Record of a single iteration."""
    
    iteration: int
    timestamp: str
    
    # Inputs
    prompt_generation: PromptGenerationResult
    
    # Inference
    stack_job_id: str
    inference_duration_seconds: float
    
    # Evaluation
    grounding_evaluation: GroundingEvaluation
    
    # Refinement (if applicable)
    refinement_suggestion: Optional[RefinementSuggestion] = None


class ExecutionLog(BaseModel):
    """Complete execution log for reproducibility."""
    
    # Metadata
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    
    # Configuration
    config: dict
    random_seed: int
    
    # Query
    raw_query: str
    structured_query: StructuredQuery
    
    # Iterations
    iterations: list[IterationRecord]
    
    # Final result
    final_score: int
    termination_reason: str
    total_iterations: int
    
    # Output paths
    output_anndata_path: str
    output_report_path: str


class FinalOutput(BaseModel):
    """Final output specification."""
    
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
    
    # Confidence
    high_confidence_predictions: list[str]
    novel_predictions: list[str]
    low_confidence_flags: list[str]
    
    # Files
    anndata_path: str
    report_path: str
    log_path: str
```

### 4.2 Configuration Models

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


class SLURMConfig(BaseModel):
    """SLURM job configuration."""
    
    partition: str = "gpu"
    gpus: int = 1
    gpu_type: str = "h100"
    cpus_per_task: int = 8
    mem_gb: int = 64
    time_limit: str = "2:00:00"
    
    # Submitit specific
    slurm_account: Optional[str] = None
    slurm_qos: Optional[str] = None


class IterationConfig(BaseModel):
    """Iteration control configuration."""
    
    max_iterations: int = 5
    score_threshold: int = 7  # 1-10 scale
    plateau_window: int = 3  # Stop if no improvement over N iterations
    min_improvement: int = 1  # Minimum score improvement to not count as plateau


class DatabaseConfig(BaseModel):
    """Biological database API configuration."""
    
    # Retry settings
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    
    # Rate limiting
    requests_per_minute: int = 30


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    
    db_path: str  # Path to LanceDB database directory
    table_name: str = "cells"  # Table containing cells with vectors and metadata
    embedding_dim: int = 1600  # STACK Large embedding dimension (100 tokens × 16 dim)
    
    # Search parameters
    default_k: int = 100
    ef_search: int = 200  # HNSW parameter


class HaystackConfig(BaseModel):
    """Complete HAYSTACK configuration."""
    
    # Core settings
    run_id: Optional[str] = None  # Auto-generated if not provided
    random_seed: int = 42
    output_dir: str = "./haystack_output"
    
    # Components
    llm: LLMConfig
    slurm: SLURMConfig
    iteration: IterationConfig
    databases: DatabaseConfig
    vector_db: VectorDBConfig
    
    # STACK model
    stack_checkpoint: str
    stack_genelist: str
    
    # Atlas paths
    parse_pbmc_path: str
    openproblems_path: str
    tabula_sapiens_path: str
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_intermediate: bool = True
```

---

## 5. Agent Specifications

### 5.1 Orchestrator Agent

The orchestrator is the main entry point, implemented using DeepAgents.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

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
        submit_stack_inference_tool,
        wait_for_job_tool,
    ],
    subagents=[
        query_understanding_subagent,
        prompt_generation_subagent,
        grounding_evaluation_subagent,
    ],
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    checkpointer=MemorySaver(),
    backend=FilesystemBackend(root_dir=config.output_dir),
)
```

**System Prompt** (summarized):
```
You are the HAYSTACK orchestrator. Your job is to:
1. Understand the user's perturbation prediction query
2. Generate biologically-informed prompts for STACK
3. Run STACK inference and evaluate biological grounding
4. Iteratively refine until convergence or max iterations

Follow this workflow:
- First, delegate to query_understanding to parse the query
- Then, delegate to prompt_generation to create candidates
- Submit STACK inference job and wait for completion
- Delegate to grounding_evaluation to assess predictions
- Check convergence criteria
- If not converged, analyze failures and loop with refinement context
- When done, generate final outputs
```

### 5.2 Query Understanding Subagent

```python
query_understanding_subagent = {
    "name": "query_understanding",
    "description": "Parse natural language queries into structured perturbation prediction requests",
    "prompt": QUERY_UNDERSTANDING_PROMPT,
    "tools": [
        resolve_cell_type_tool,      # Maps to Cell Ontology
        resolve_perturbation_tool,   # Maps to ChEBI/DrugBank
        get_drug_targets_tool,       # KEGG, DrugBank
        get_pathway_info_tool,       # KEGG, Reactome
        assess_feasibility_tool,     # Check against available data
    ],
}
```

**Responsibilities**:
1. Parse intent from natural language
2. Resolve cell type to Cell Ontology ID
3. Resolve perturbation to external database ID
4. Retrieve known targets and expected pathways
5. Assess feasibility given available prompt data

### 5.3 Prompt Generation Subagent

```python
prompt_generation_subagent = {
    "name": "prompt_generation",
    "description": "Generate and rank prompt candidates using multiple strategies",
    "prompt": PROMPT_GENERATION_PROMPT,
    "tools": [
        # Strategy tools (run in parallel)
        direct_match_search_tool,
        mechanistic_match_search_tool,
        semantic_vector_search_tool,
        ontology_guided_search_tool,
        
        # Ranking
        rank_candidates_tool,
    ],
}
```

**Prompt Generation Strategies**:

| Strategy | Description | Tools Used |
|----------|-------------|------------|
| Direct Match | Find exact perturbation in available datasets | Atlas metadata search |
| Mechanistic Match | Find perturbations sharing targets/pathways | KEGG, Reactome, DrugBank |
| Semantic Match | Find similar cell states via embeddings | Vector DB (LanceDB) |
| Ontology-Guided | Select context cells by lineage relationships | Cell Ontology |

### 5.4 Grounding Evaluation Subagent

```python
grounding_evaluation_subagent = {
    "name": "grounding_evaluation",
    "description": "Evaluate biological plausibility of STACK predictions",
    "prompt": GROUNDING_EVALUATION_PROMPT,
    "tools": [
        # Enrichment analysis
        go_enrichment_tool,
        kegg_enrichment_tool,
        reactome_enrichment_tool,
        
        # Network analysis
        string_ppi_tool,
        
        # Literature
        pubmed_search_tool,
        extract_evidence_tool,
        
        # Scoring
        compute_pathway_score_tool,
        compute_target_score_tool,
        compute_literature_score_tool,
        compute_network_score_tool,
        compute_composite_score_tool,
    ],
}
```

**Scoring Components**:

| Component | Score Range | Criteria |
|-----------|-------------|----------|
| Pathway Coherence | 1-10 | Expected pathways enriched? Unexpected pathways absent? |
| Target Activation | 1-10 | Known targets differentially expressed in expected direction? |
| Literature Support | 1-10 | DE genes supported by published evidence? |
| Network Coherence | 1-10 | DE genes form connected network with targets? |
| **Composite** | 1-10 | Weighted average with asymmetric penalties |

**Composite Score Calculation**:
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

## 6. Tool Specifications

### 6.1 Drug-Target Knowledge Tools

```python
@tool
def get_drug_targets(
    perturbation: str,
    perturbation_type: str,
) -> dict:
    """
    Retrieve known targets for a perturbation.
    
    Args:
        perturbation: Name of drug/cytokine/gene
        perturbation_type: One of 'drug', 'cytokine', 'genetic'
    
    Returns:
        Dictionary with:
        - targets: List of gene symbols
        - target_types: List of target types (e.g., 'receptor', 'enzyme')
        - sources: List of database sources
        - confidence: Confidence level
    
    Databases:
        - KEGG DRUG (for drugs)
        - UniProt (for receptors/binding partners)
        - Reactome (for signaling components)
    """
    ...


@tool
def get_pathway_memberships(
    genes: list[str],
) -> dict:
    """
    Get pathway memberships for a list of genes.
    
    Args:
        genes: List of gene symbols
    
    Returns:
        Dictionary with:
        - kegg_pathways: Dict mapping pathway ID to pathway name
        - reactome_pathways: Dict mapping pathway ID to pathway name  
        - go_terms: Dict mapping GO ID to term name
    
    Databases:
        - KEGG API
        - Reactome API
        - Gene Ontology API
    """
    ...


@tool
def find_mechanistically_similar_perturbations(
    target_genes: list[str],
    pathways: list[str],
    available_perturbations: list[str],
) -> list[dict]:
    """
    Find perturbations that share targets or pathways with query.
    
    Args:
        target_genes: Genes targeted by query perturbation
        pathways: Pathways affected by query perturbation
        available_perturbations: Perturbations available in prompt datasets
    
    Returns:
        List of dicts with:
        - perturbation: Name
        - overlap_score: Jaccard similarity of targets/pathways
        - shared_targets: List of shared target genes
        - shared_pathways: List of shared pathways
    """
    ...
```

### 6.2 Cell Ontology Tools

```python
@tool
def resolve_cell_type(
    cell_type_name: str,
) -> dict:
    """
    Resolve cell type name to Cell Ontology ID.
    
    Args:
        cell_type_name: Free-text cell type name
    
    Returns:
        Dictionary with:
        - cl_id: Cell Ontology ID (e.g., CL:0000235)
        - canonical_name: Canonical name from CL
        - synonyms: List of synonyms
        - parent_types: List of parent cell types
        - lineage: Lineage path to root
    """
    ...


@tool
def get_related_cell_types(
    cl_id: str,
    relationship: str = "is_a",
    max_distance: int = 2,
) -> list[dict]:
    """
    Get cell types related to query cell type.
    
    Args:
        cl_id: Cell Ontology ID
        relationship: Relationship type ('is_a', 'develops_from', 'part_of')
        max_distance: Maximum ontology distance
    
    Returns:
        List of related cell types with CL IDs and distances
    """
    ...


@tool
def find_cell_type_in_atlas(
    cl_id: str,
    atlas: str,
) -> dict:
    """
    Find cells matching a cell type in an atlas.
    
    Args:
        cl_id: Cell Ontology ID
        atlas: One of 'parse_pbmc', 'openproblems', 'tabula_sapiens'
    
    Returns:
        Dictionary with:
        - found: Boolean
        - cell_type_column: Column name in atlas
        - matching_labels: List of matching cell type labels
        - cell_count: Number of matching cells
        - indices: Cell indices (if found)
    """
    ...
```

### 6.3 Vector Database Tools

```python
@tool
def search_similar_cells(
    query_embedding: list[float],
    k: int = 100,
    filters: dict = None,
) -> list[dict]:
    """
    Search for cells with similar STACK embeddings.
    
    Args:
        query_embedding: STACK embedding vector (dim=1600)
        k: Number of results
        filters: Optional filters (cell_type, tissue, perturbation, dataset)
    
    Returns:
        List of cell matches with:
        - cell_id: Unique cell identifier
        - distance: L2 distance to query
        - cell_type: Cell type annotation
        - tissue: Tissue of origin
        - perturbation: Perturbation status (if any)
        - dataset: Source dataset
        - index: Index into atlas file
    """
    ...


@tool
def get_embedding_for_cell_set(
    indices: list[int],
    dataset: str,
) -> list[float]:
    """
    Get mean STACK embedding for a set of cells.
    
    Args:
        indices: Cell indices
        dataset: Source dataset
    
    Returns:
        Mean embedding vector (dim=1600)
    """
    ...


@tool
def find_perturbed_cells_by_condition(
    perturbation: str,
    dataset: str = None,
) -> list[dict]:
    """
    Find cells that were experimentally perturbed with a condition.
    
    Args:
        perturbation: Perturbation name
        dataset: Optionally restrict to specific dataset
    
    Returns:
        List of cell groups with indices and metadata
    """
    ...
```

### 6.4 SLURM/STACK Inference Tools

```python
@tool
def submit_stack_inference(
    prompt_cell_indices: list[int],
    prompt_dataset: str,
    query_cell_indices: list[int],
    query_dataset: str,
    context_cell_indices: list[int],
    context_dataset: str,
    output_path: str,
    config: dict,
) -> dict:
    """
    Submit STACK inference job to SLURM.
    
    Args:
        prompt_cell_indices: Indices of prompt cells
        prompt_dataset: Dataset containing prompt cells
        query_cell_indices: Indices of query cells
        query_dataset: Dataset containing query cells
        context_cell_indices: Additional context cells
        context_dataset: Dataset containing context cells
        output_path: Path for output AnnData
        config: SLURM and STACK configuration
    
    Returns:
        Dictionary with:
        - job_id: SLURM job ID
        - status: 'submitted', 'running', 'completed', 'failed'
        - output_path: Path to results when complete
    
    Implementation:
        Uses submitit to submit job to SLURM GPU partition
    """
    ...


@tool
def wait_for_job(
    job_id: str,
    timeout_seconds: int = 7200,
    poll_interval: int = 30,
) -> dict:
    """
    Wait for SLURM job to complete.
    
    Args:
        job_id: SLURM job ID
        timeout_seconds: Maximum wait time
        poll_interval: Seconds between status checks
    
    Returns:
        Dictionary with:
        - status: Final status
        - duration_seconds: Total runtime
        - output_path: Path to results (if successful)
        - error: Error message (if failed)
    """
    ...


@tool
def extract_de_genes(
    prediction_path: str,
    control_path: str,
    lfc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
) -> list[dict]:
    """
    Extract differentially expressed genes from predictions.
    
    Args:
        prediction_path: Path to prediction AnnData
        control_path: Path to control/baseline AnnData
        lfc_threshold: Minimum log2 fold change
        pval_threshold: Maximum adjusted p-value
    
    Returns:
        List of DE genes with statistics
    """
    ...
```

### 6.5 Enrichment Analysis Tools

```python
@tool
def run_go_enrichment(
    genes: list[str],
    background_genes: list[str] = None,
    ontology: str = "BP",
) -> dict:
    """
    Run Gene Ontology enrichment analysis.
    
    Args:
        genes: List of DE gene symbols
        background_genes: Background gene set (default: all expressed genes)
        ontology: 'BP' (biological process), 'MF' (molecular function), 'CC' (cellular component)
    
    Returns:
        Dictionary with:
        - enriched_terms: List of significant GO terms
        - p_values: Dict of term -> p-value
        - fold_enrichments: Dict of term -> fold enrichment
        - genes_per_term: Dict of term -> genes
    """
    ...


@tool
def run_kegg_enrichment(
    genes: list[str],
    organism: str = "hsa",
) -> dict:
    """
    Run KEGG pathway enrichment analysis.
    
    Args:
        genes: List of DE gene symbols
        organism: KEGG organism code
    
    Returns:
        Dictionary with enriched KEGG pathways
    """
    ...


@tool  
def run_reactome_enrichment(
    genes: list[str],
) -> dict:
    """
    Run Reactome pathway enrichment analysis.
    
    Args:
        genes: List of DE gene symbols
    
    Returns:
        Dictionary with enriched Reactome pathways
    """
    ...
```

### 6.6 Literature Search Tools

```python
@tool
def search_pubmed(
    query: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Search PubMed for relevant papers.
    
    Args:
        query: Search query (e.g., "IFN-gamma STAT1 macrophage")
        max_results: Maximum papers to return
    
    Returns:
        List of papers with:
        - pmid: PubMed ID
        - title: Paper title
        - abstract: Abstract text
        - year: Publication year
        - journal: Journal name
    """
    ...


@tool
def extract_gene_evidence(
    papers: list[dict],
    gene: str,
    perturbation: str,
) -> dict:
    """
    Extract evidence about a gene from papers.
    
    Args:
        papers: List of paper dictionaries
        gene: Gene symbol
        perturbation: Perturbation of interest
    
    Returns:
        Dictionary with:
        - has_evidence: Boolean
        - supports_upregulation: Boolean or None
        - supports_downregulation: Boolean or None
        - contradicts: Boolean
        - citations: List of supporting PMIDs
        - snippets: Relevant text excerpts
    """
    ...
```

### 6.7 Network Analysis Tools

```python
@tool
def get_ppi_network(
    genes: list[str],
    score_threshold: int = 400,
) -> dict:
    """
    Get protein-protein interaction network from STRING.
    
    Args:
        genes: List of gene symbols
        score_threshold: Minimum interaction score (1-1000)
    
    Returns:
        Dictionary with:
        - nodes: List of genes in network
        - edges: List of (gene1, gene2, score) tuples
        - connected_components: Number of connected components
        - largest_component_size: Size of largest component
    """
    ...


@tool
def compute_network_connectivity(
    de_genes: list[str],
    target_genes: list[str],
) -> dict:
    """
    Compute connectivity between DE genes and targets.
    
    Args:
        de_genes: Differentially expressed genes
        target_genes: Known target genes
    
    Returns:
        Dictionary with:
        - direct_connections: DE genes directly connected to targets
        - indirect_connections: DE genes within 2 hops of targets
        - connectivity_score: Overall connectivity metric (1-10)
    """
    ...
```

---

## 7. Biological Database Integration

### 7.1 Databases Used (MVP)

| Database | Purpose | Access Method | Rate Limit |
|----------|---------|---------------|------------|
| **KEGG** | Pathways, drug targets | REST API | 10 req/sec |
| **Reactome** | Pathway analysis | REST API | None specified |
| **STRING** | PPI network | REST API | 1 req/sec |
| **Gene Ontology** | Enrichment, ontology | REST API | None specified |
| **UniProt** | Protein info, targets | REST API | 100 req/min |
| **PubMed** | Literature search | E-utilities | 3 req/sec |
| **Cell Ontology** | Cell type resolution | OBO file (local) | N/A |

### 7.2 API Client Pattern

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class BiologicalDatabaseClient:
    """Base class for biological database API clients."""
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """
        Initialize database client.
        
        Args:
            base_url: Base URL for API
            max_retries: Maximum retry attempts
            base_delay: Initial delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.client = httpx.Client(timeout=30.0)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with retry and exponential backoff."""
        response = self.client.get(
            f"{self.base_url}/{endpoint}",
            params=params,
        )
        response.raise_for_status()
        return response.json()
```

### 7.3 Database-Specific Implementations

```python
class KEGGClient(BiologicalDatabaseClient):
    """KEGG API client."""
    
    def __init__(self):
        super().__init__(base_url="https://rest.kegg.jp")
    
    def get_pathway_genes(self, pathway_id: str) -> list[str]:
        """Get genes in a KEGG pathway."""
        ...
    
    def get_drug_targets(self, drug_id: str) -> list[str]:
        """Get targets for a KEGG drug."""
        ...
    
    def find_pathways_for_genes(self, genes: list[str]) -> dict:
        """Find pathways containing given genes."""
        ...


class ReactomeClient(BiologicalDatabaseClient):
    """Reactome API client."""
    
    def __init__(self):
        super().__init__(base_url="https://reactome.org/ContentService")
    
    def run_enrichment(self, genes: list[str]) -> dict:
        """Run pathway enrichment analysis."""
        ...
    
    def get_pathway_hierarchy(self, pathway_id: str) -> dict:
        """Get pathway and its parent pathways."""
        ...


class STRINGClient(BiologicalDatabaseClient):
    """STRING API client."""
    
    def __init__(self):
        super().__init__(base_url="https://string-db.org/api")
    
    def get_interactions(
        self, 
        genes: list[str], 
        score_threshold: int = 400,
    ) -> list[dict]:
        """Get protein-protein interactions."""
        ...
    
    def get_enrichment(self, genes: list[str]) -> dict:
        """Get functional enrichment from STRING."""
        ...
```

---

## 8. Vector Database Specification

### 8.1 Index Structure

```python
import lancedb

class STACKVectorIndex:
    """
    LanceDB-based vector index for STACK embeddings with integrated metadata.
    
    LanceDB stores vectors and metadata together in a single table, supporting
    efficient filtered vector search via scalar indexes on metadata columns.
    
    Indexed Atlases:
    - Parse PBMC (12 donors, 90 cytokine perturbations, ~10M cells)
    - OpenProblems (3 donors, 147 drug conditions, ~500K cells)
    - Tabula Sapiens (24 donors, 25 tissues, ~500K cells after balancing)
    """
    
    def __init__(
        self,
        db_path: str,
        table_name: str = "cells",
        embedding_dim: int = 1600,
    ):
        """
        Initialize vector index.
        
        Args:
            db_path: Path to LanceDB database directory
            table_name: Name of the table containing cells
            embedding_dim: STACK embedding dimension (100 tokens × 16 dim = 1600)
        """
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.embedding_dim = embedding_dim
    
    def search(
        self,
        query_vector: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Search for similar cells with optional metadata filtering.
        
        Args:
            query_vector: STACK embedding vector
            k: Number of results to return
            filters: Optional dict of column -> value filters
        
        Returns:
            List of matching cells with vectors and metadata
        """
        query = self.table.search(query_vector).limit(k)
        
        if filters:
            # Build SQL-style filter string
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                elif isinstance(val, list):
                    vals = ", ".join(f"'{v}'" for v in val)
                    conditions.append(f"{col} IN ({vals})")
                else:
                    conditions.append(f"{col} = {val}")
            query = query.where(" AND ".join(conditions))
        
        return query.to_list()
```

### 8.2 Table Schema

LanceDB stores vectors and metadata together. The table schema includes both the embedding vector and all cell annotations:

```python
# Schema for the cells table
# LanceDB infers types from the data, but conceptually:
schema = {
    # Vector column (automatically indexed for ANN search)
    "vector": list[float],  # FixedSizeList[Float32, 1600]
    
    # Cell identifiers
    "cell_id": int,         # Unique cell identifier
    "dataset": str,         # 'parse_pbmc', 'openproblems', 'tabula_sapiens'
    "atlas_index": int,     # Index in source H5AD file
    
    # Annotations
    "cell_type": str,
    "cell_type_cl_id": str,  # Cell Ontology ID
    "tissue": str,
    "donor_id": str,
    
    # Perturbation status
    "is_perturbed": bool,
    "perturbation": str,
    "perturbation_type": str,  # 'drug', 'cytokine', 'genetic', None
    "control_status": str,     # 'control', 'perturbed', None
    
    # Quality metrics
    "n_genes": int,
    "total_counts": float,
}

# Create scalar indexes on frequently filtered columns for performance
table.create_scalar_index("dataset")
table.create_scalar_index("cell_type")
table.create_scalar_index("perturbation")
table.create_scalar_index("tissue")
table.create_scalar_index("is_perturbed", index_type="BITMAP")  # Low cardinality
```

### 8.3 Index Building (One-time Setup)

```python
def build_stack_vector_index(
    parse_path: str,
    openproblems_path: str,
    tabula_sapiens_path: str,
    stack_checkpoint: str,
    output_db_path: str,
    table_name: str = "cells",
):
    """
    Build LanceDB index from atlas embeddings.
    
    This is a one-time setup step, not part of runtime.
    Requires GPU for STACK embedding computation.
    
    LanceDB stores vectors and metadata together in a single table,
    so only one output path is needed.
    
    Args:
        parse_path: Path to Parse PBMC H5AD
        openproblems_path: Path to OpenProblems H5AD
        tabula_sapiens_path: Path to Tabula Sapiens H5AD
        stack_checkpoint: Path to STACK model checkpoint
        output_db_path: Path for output LanceDB database directory
        table_name: Name of the table to create
    """
    ...
```

---

## 9. SLURM Integration

### 9.1 Submitit Configuration

```python
import submitit

class SLURMJobManager:
    """Manages SLURM job submission and monitoring using submitit."""
    
    def __init__(self, config: SLURMConfig):
        """
        Initialize job manager.
        
        Args:
            config: SLURM configuration
        """
        self.config = config
        self.executor = submitit.AutoExecutor(folder="./slurm_logs")
        self.executor.update_parameters(
            slurm_partition=config.partition,
            slurm_gpus_per_node=config.gpus,
            slurm_cpus_per_task=config.cpus_per_task,
            slurm_mem_gb=config.mem_gb,
            slurm_time=config.time_limit,
            slurm_constraint=config.gpu_type,  # e.g., "h100"
            slurm_account=config.slurm_account,
            slurm_qos=config.slurm_qos,
        )
    
    def submit_stack_job(
        self,
        prompt_adata_path: str,
        query_adata_path: str,
        output_path: str,
        stack_config: dict,
    ) -> submitit.Job:
        """Submit STACK inference job."""
        job = self.executor.submit(
            run_stack_inference,
            prompt_adata_path=prompt_adata_path,
            query_adata_path=query_adata_path,
            output_path=output_path,
            config=stack_config,
        )
        return job
    
    def wait_for_job(
        self,
        job: submitit.Job,
        timeout: int = 7200,
    ) -> dict:
        """Wait for job completion and return results."""
        try:
            result = job.result(timeout=timeout)
            return {
                "status": "completed",
                "result": result,
                "duration": job.get_info().get("runtime", 0),
            }
        except submitit.core.utils.FailedJobError as e:
            return {
                "status": "failed",
                "error": str(e),
            }
        except TimeoutError:
            return {
                "status": "timeout",
                "error": f"Job exceeded {timeout}s timeout",
            }
```

### 9.2 STACK Inference Function

```python
def run_stack_inference(
    prompt_adata_path: str,
    query_adata_path: str,
    output_path: str,
    config: dict,
) -> str:
    """
    Run STACK inference on GPU node.
    
    This function is submitted to SLURM and runs on GPU.
    
    Args:
        prompt_adata_path: Path to prompt cells AnnData
        query_adata_path: Path to query cells AnnData
        output_path: Path for output predictions
        config: STACK configuration dict
    
    Returns:
        Path to output file
    """
    import torch
    import scanpy as sc
    from stack.inference import STACKPredictor
    
    # Load STACK model
    predictor = STACKPredictor(
        checkpoint=config["checkpoint"],
        genelist=config["genelist"],
        device="cuda",
    )
    
    # Load data
    prompt_adata = sc.read_h5ad(prompt_adata_path)
    query_adata = sc.read_h5ad(query_adata_path)
    
    # Run inference with mask diffusion (T=5)
    predictions = predictor.predict(
        prompt_adata=prompt_adata,
        query_adata=query_adata,
        T=config.get("diffusion_steps", 5),
        batch_size=config.get("batch_size", 32),
    )
    
    # Save results
    predictions.write_h5ad(output_path)
    
    return output_path
```

### 9.3 Job Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SLURM JOB WORKFLOW                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  CPU Node (Orchestrator)                                                        │
│  │                                                                              │
│  │  1. Prepare cell sets                                                        │
│  │     - Extract prompt cells from atlas                                        │
│  │     - Extract query cells from atlas                                         │
│  │     - Save to temporary H5AD files                                           │
│  │                                                                              │
│  │  2. Submit job via submitit                                                  │
│  │     └─────────────────────────────────────────────────────┐                  │
│  │                                                           │                  │
│  │                                                           ▼                  │
│  │                                           ┌───────────────────────────┐      │
│  │                                           │   GPU Node (H100)         │      │
│  │                                           │                           │      │
│  │                                           │   3. Load STACK model     │      │
│  │                                           │   4. Load cell data       │      │
│  │                                           │   5. Run inference (T=5)  │      │
│  │                                           │   6. Save predictions     │      │
│  │                                           │                           │      │
│  │                                           └───────────────────────────┘      │
│  │                                                           │                  │
│  │  7. Poll for completion                                   │                  │
│  │     ◄─────────────────────────────────────────────────────┘                  │
│  │                                                                              │
│  │  8. Load results and continue                                                │
│  │                                                                              │
│  ▼                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. CLI Interface

### 10.1 Command Structure

```bash
# Main command
haystack <query> [options]

# With config file
haystack <query> --config config.yaml

# Examples
haystack "How would lung fibroblasts respond to TGF-beta treatment?"

haystack "Predict the effect of dexamethasone on alveolar macrophages" \
    --model anthropic:claude-sonnet-4-5-20250929 \
    --max-iterations 5 \
    --score-threshold 7 \
    --output-dir ./results
```

### 10.2 CLI Arguments

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="HAYSTACK: Iterative Knowledge-Guided Cell Prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        "query",
        type=str,
        help="Natural language query describing the prediction task",
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    
    # LLM settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="anthropic:claude-sonnet-4-5-20250929",
        help="LLM model (format: provider:model_name)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature",
    )
    
    # Iteration control
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=7,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Grounding score threshold for convergence (1-10)",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=3,
        help="Stop if no improvement over N iterations",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./haystack_output",
        help="Output directory",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run ID (auto-generated if not provided)",
    )
    
    # SLURM
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="gpu",
        help="SLURM partition for GPU jobs",
    )
    parser.add_argument(
        "--slurm-account",
        type=str,
        help="SLURM account",
    )
    parser.add_argument(
        "--slurm-qos",
        type=str,
        help="SLURM QoS",
    )
    
    # Data paths
    parser.add_argument(
        "--stack-checkpoint",
        type=str,
        required=True,
        help="Path to STACK model checkpoint",
    )
    parser.add_argument(
        "--stack-genelist",
        type=str,
        required=True,
        help="Path to STACK gene list",
    )
    parser.add_argument(
        "--vector-index",
        type=str,
        required=True,
        help="Path to LanceDB vector index",
    )
    parser.add_argument(
        "--parse-pbmc",
        type=str,
        required=True,
        help="Path to Parse PBMC atlas",
    )
    parser.add_argument(
        "--openproblems",
        type=str,
        required=True,
        help="Path to OpenProblems atlas",
    )
    parser.add_argument(
        "--tabula-sapiens",
        type=str,
        required=True,
        help="Path to Tabula Sapiens atlas",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()
```

### 10.3 Entry Point

```python
def main():
    """Main entry point for haystack CLI."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.verbose)
    
    # Load/build configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = build_config_from_args(args)
    
    # Generate run ID if not provided
    if not config.run_id:
        config.run_id = generate_run_id()
    
    # Initialize and run
    logger.info(f"Starting HAYSTACK run: {config.run_id}")
    logger.info(f"Query: {args.query}")
    
    orchestrator = HaystackOrchestrator(config)
    result = orchestrator.run(args.query)
    
    # Print summary
    print_result_summary(result)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## 11. Configuration

### 11.1 Configuration File Format (YAML)

```yaml
# haystack_config.yaml

# Run settings
run_id: null  # Auto-generated if null
random_seed: 42
output_dir: ./haystack_output

# LLM configuration
llm:
  provider: anthropic  # anthropic, openai, google_genai
  model: claude-sonnet-4-5-20250929
  temperature: 0.7
  max_tokens: 4096
  api_key_env_var: ANTHROPIC_API_KEY

# SLURM configuration
slurm:
  partition: gpu
  gpus: 1
  gpu_type: h100
  cpus_per_task: 8
  mem_gb: 64
  time_limit: "2:00:00"
  account: null
  qos: null

# Iteration control
iteration:
  max_iterations: 5
  score_threshold: 7  # 1-10 scale
  plateau_window: 3
  min_improvement: 1

# Database API settings
databases:
  max_retries: 3
  base_delay_seconds: 1.0
  max_delay_seconds: 30.0
  requests_per_minute: 30

# Vector database (LanceDB stores vectors + metadata together)
vector_db:
  db_path: /path/to/stack_embeddings.lancedb
  table_name: cells
  embedding_dim: 1600
  default_k: 100
  ef_search: 200

# STACK model
stack:
  checkpoint: /path/to/stack_large_aligned.ckpt
  genelist: /path/to/hvg_genes.pkl
  diffusion_steps: 5
  batch_size: 32

# Atlas paths
atlases:
  parse_pbmc: /path/to/parse_pbmc.h5ad
  openproblems: /path/to/openproblems.h5ad
  tabula_sapiens: /path/to/tabula_sapiens.h5ad

# Logging
logging:
  level: INFO
  save_intermediate: true
```

### 11.2 Environment Variables

```bash
# Required API keys (at least one)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# Optional SLURM settings
export SLURM_ACCOUNT="my_account"
export SLURM_QOS="normal"

# Optional paths
export HAYSTACK_CONFIG="/path/to/default_config.yaml"
```

---

## 12. Output Specification

### 12.1 Output Directory Structure

```
haystack_output/
├── {run_id}/
│   ├── predictions.h5ad          # Final predicted expression
│   ├── report.md                 # Human-readable interpretation
│   ├── report.html               # HTML version of report
│   ├── execution_log.json        # Complete execution log
│   └── iterations/
│       ├── iter_001/
│       │   ├── prompt_cells.h5ad
│       │   ├── query_cells.h5ad
│       │   ├── predictions.h5ad
│       │   ├── evaluation.json
│       │   └── slurm_log.txt
│       ├── iter_002/
│       │   └── ...
│       └── ...
```

### 12.2 AnnData Output Schema

```python
# predictions.h5ad structure

adata.X                    # Predicted expression matrix (cells x genes)
adata.obs                  # Cell metadata
    - cell_id              # Unique cell identifier
    - original_cell_type   # Original cell type from query
    - predicted_state      # 'perturbed' 
    - confidence_score     # MLP classifier confidence
    - iteration            # Which iteration produced this prediction

adata.var                  # Gene metadata
    - gene_symbol          # HUGO gene symbol
    - is_de                # Boolean: differentially expressed
    - log2_fold_change     # Predicted LFC vs control
    - p_value              # Statistical significance
    - adjusted_p_value     # BH-adjusted p-value
    - direction            # 'up' or 'down'
    - in_expected_targets  # Boolean: known target gene
    - pathway_annotations  # List of pathway memberships
    - literature_support   # 'supported', 'contradicted', 'novel', None

adata.uns                  # Unstructured metadata
    - query                # Original query
    - structured_query     # Parsed query dict
    - prompt_info          # Information about selected prompt
    - grounding_scores     # Final evaluation scores
    - enrichment_results   # GO/KEGG/Reactome results
    - run_id               # Run identifier
    - config               # Configuration used
```

### 12.3 Report Format

```markdown
# HAYSTACK Prediction Report

## Query
**Original Query**: "How would lung fibroblasts respond to TGF-beta treatment?"

**Parsed Query**:
- Cell Type: Lung fibroblast (CL:0002553)
- Perturbation: TGF-β (cytokine)
- Expected Pathways: TGF-beta signaling, EMT, ECM organization

## Results Summary

**Grounding Score**: 8/10 ✓
**Convergence**: Achieved at iteration 3
**Termination Reason**: Score threshold reached

## Prediction Highlights

### Top Upregulated Genes
| Gene | Log2FC | P-value | Known Target | Pathway |
|------|--------|---------|--------------|---------|
| COL1A1 | 2.3 | 1e-10 | Yes | ECM organization |
| ACTA2 | 1.9 | 1e-8 | Yes | Smooth muscle contraction |
| ... | ... | ... | ... | ... |

### Top Downregulated Genes
| Gene | Log2FC | P-value | Pathway |
|------|--------|---------|---------|
| ... | ... | ... | ... |

### Enriched Pathways
1. TGF-beta signaling pathway (KEGG:hsa04350) - p < 1e-15
2. ECM-receptor interaction (KEGG:hsa04512) - p < 1e-12
3. ...

## Confidence Assessment

**High Confidence Predictions**: 45 genes
- Supported by literature and pathway analysis

**Novel Predictions**: 12 genes
- No literature support but consistent with network analysis
- Potential new biology - consider for follow-up

**Low Confidence Flags**:
- Receptor genes (TGFBR1, TGFBR2) not differentially expressed
- Consider: TGF-β prompt may not capture receptor dynamics

## Iteration History

| Iteration | Score | Best Strategy | Key Improvement |
|-----------|-------|---------------|-----------------|
| 1 | 5 | Direct match | - |
| 2 | 7 | Mechanistic | Better pathway coherence |
| 3 | 8 | Refined semantic | Improved literature support |

## Files

- Predictions: `predictions.h5ad`
- Full Log: `execution_log.json`

---
*Generated by HAYSTACK v0.1.0*
*Run ID: abc123*
*Timestamp: 2026-01-15T10:30:00Z*
```

---

## 13. Error Handling

### 13.1 Error Categories

| Category | Examples | Handling Strategy |
|----------|----------|-------------------|
| **Query Errors** | Unparseable query, unknown cell type | Return error with suggestions |
| **Data Errors** | Missing atlas, corrupted files | Fail fast with clear message |
| **API Errors** | Rate limits, timeouts | Retry with exponential backoff |
| **SLURM Errors** | Job failure, timeout | Capture logs, allow retry |
| **Convergence Failures** | Max iterations, plateau | Return best result with warning |

### 13.2 Error Response Model

```python
class HaystackError(BaseModel):
    """Structured error response."""
    
    error_type: str
    message: str
    details: dict = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    recoverable: bool = False
    partial_results: Optional[dict] = None
```

### 13.3 Graceful Degradation

```python
def handle_api_failure(error: Exception, tool_name: str) -> dict:
    """
    Handle API failures gracefully.
    
    Args:
        error: The exception that occurred
        tool_name: Name of the tool that failed
    
    Returns:
        Dictionary with fallback behavior
    
    Strategy:
    1. For non-critical tools (e.g., literature), continue without
    2. For critical tools (e.g., KEGG), retry then fail
    3. Always log and report degraded functionality
    """
    ...
```

---

## 14. Testing Strategy

### 14.1 Test Categories

| Category | Scope | Location |
|----------|-------|----------|
| Unit Tests | Individual tools, utilities | `tests/unit/` |
| Integration Tests | Agent workflows, API integration | `tests/integration/` |
| End-to-End Tests | Full pipeline on test queries | `tests/e2e/` |
| Benchmark Tests | Performance, accuracy metrics | `tests/benchmark/` |

### 14.2 Test Fixtures

```python
# Test queries with known expected outcomes
TEST_QUERIES = [
    {
        "query": "Predict IFN-gamma effect on macrophages",
        "expected_pathways": ["Interferon signaling", "JAK-STAT"],
        "expected_targets": ["STAT1", "IRF1", "GBP1"],
        "min_score": 6,
    },
    {
        "query": "How would T cells respond to IL-2 stimulation",
        "expected_pathways": ["IL-2 signaling", "T cell activation"],
        "expected_targets": ["IL2RA", "STAT5A", "STAT5B"],
        "min_score": 6,
    },
    # ... more test cases
]
```

### 14.3 Mock Services

```python
class MockKEGGClient(KEGGClient):
    """Mock KEGG client for testing without API calls."""
    
    def __init__(self, fixture_path: str):
        self.fixtures = load_fixtures(fixture_path)
    
    def get_pathway_genes(self, pathway_id: str) -> list[str]:
        return self.fixtures.get(pathway_id, {}).get("genes", [])
```

---

## 15. Dependencies

### 15.1 Python Dependencies

```toml
[project]
name = "haystack"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
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
    
    # Vector database
    "lancedb>=0.5.0",
    
    # SLURM integration
    "submitit>=1.5.0",
    
    # API clients
    "httpx>=0.25.0",
    "tenacity>=8.0.0",
    
    # Utilities
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
    
    # Biological analysis
    "gseapy>=1.0.0",
    "networkx>=3.0.0",
    
    # Testing
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]
```

### 15.2 External Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| STACK | Foundation model | `pip install arc-stack` |
| SLURM | Job scheduler | Cluster installation |
| CUDA | GPU support | System installation |

---

## 16. Future Extensions

### 16.1 Post-MVP Features

| Feature | Priority | Description |
|---------|----------|-------------|
| User custom atlases | High | Allow users to add their own H5AD files |
| Results caching | High | SQLite cache for API responses and enrichments |
| Interactive mode | Medium | Real-time feedback during iteration |
| Multi-query batching | Medium | Process multiple queries efficiently |
| Validation agent | Low | Automatic comparison to experimental data |
| Fine-tuning support | Low | Improve STACK with user feedback |

### 16.2 Scalability Improvements

- Parallel prompt evaluation (multiple SLURM jobs)
- Distributed vector index (multiple shards)
- Asynchronous API calls
- Result streaming

### 16.3 Integration Opportunities

- Integration with Benchling for experimental tracking
- Integration with GEO for validation data retrieval
- Web interface for non-CLI users
- Jupyter notebook integration

---

## Appendix A: System Prompts

### A.1 Orchestrator System Prompt

```
You are the HAYSTACK orchestrator, an AI system that improves single-cell perturbation predictions through iterative knowledge-guided prompting.

Your workflow:
1. UNDERSTAND: Parse the user's query to identify cell type, perturbation, and expected biology
2. GENERATE: Create multiple prompt candidates using different strategies
3. INFER: Submit STACK inference job and wait for completion
4. EVALUATE: Assess biological grounding of predictions (1-10 score)
5. REFINE: If score < threshold and iterations remain, diagnose issues and refine

Key principles:
- Use biological knowledge to guide prompt selection
- Evaluate predictions against pathways, literature, and networks
- Distinguish "novel" predictions from "wrong" predictions
- Provide interpretable explanations for all decisions

Convergence criteria:
- Score >= {threshold}: Success
- Iterations >= {max_iterations}: Return best result
- No improvement over {plateau_window} iterations: Return current result
```

### A.2 Query Understanding Prompt

```
You are analyzing a natural language query about single-cell perturbation prediction.

Your task:
1. Identify the cell type (resolve to Cell Ontology ID if possible)
2. Identify the perturbation (drug, cytokine, or genetic)
3. Retrieve known targets and expected pathways
4. Assess feasibility given available prompt data

Output a structured query with all relevant biological context.

Available prompt datasets:
- Parse PBMC: 90 cytokine perturbations in immune cells (12 donors)
- OpenProblems: 147 drug conditions in PBMCs (3 donors)
- Tabula Sapiens: Unperturbed cells from 25 tissues (24 donors)
```

### A.3 Prompt Generation Prompt

```
You are generating prompt candidates for STACK in-context learning.

Available strategies:
1. DIRECT: Find exact perturbation match in available data
2. MECHANISTIC: Find perturbations sharing targets/pathways
3. SEMANTIC: Find similar cell states via embedding similarity
4. ONTOLOGY: Use cell type relationships to select context

For each strategy:
- Generate candidates
- Score relevance (1-10)
- Provide rationale

If refinement context is provided:
- Apply suggested modifications
- Avoid previously failed strategies
- Focus on addressing identified issues
```

### A.4 Grounding Evaluation Prompt

```
You are evaluating biological grounding of STACK predictions.

Score each component 1-10:
1. PATHWAY COHERENCE: Are expected pathways enriched?
2. TARGET ACTIVATION: Are known targets differentially expressed?
3. LITERATURE SUPPORT: Do predictions match published evidence?
4. NETWORK COHERENCE: Do DE genes connect to targets in PPI network?

Important distinctions:
- "Novel" predictions (no literature) are NOT failures
- Contradicted predictions (opposite of literature) ARE penalties
- Missing expected genes are diagnostic, not necessarily failures

Provide:
- Component scores with rationale
- Composite score (1-10)
- Diagnostics for refinement if score < threshold
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Cell Prompting** | STACK's in-context learning approach using prompt cells to condition predictions |
| **Grounding Score** | Composite measure of biological plausibility (1-10) |
| **HAYSTACK** | **H**euristic **A**gent for **Y**ielding **S**TACK-**T**uned **A**ssessments with **C**losed-loop **K**nowledge |
| **Mechanistic Match** | Prompt selection based on shared drug targets or pathways |
| **Perturb Sapiens** | STACK-generated atlas of 201 perturbations across 28 tissues and 40 cell classes |
| **Semantic Match** | Prompt selection based on embedding similarity |
| **STACK** | Single-cell foundation model using tabular attention for in-context learning |

---

## Appendix C: Changelog from v1

### Changes Made

1. **Removed Results Cache (NG6)**
   - Removed SQLite cache for enrichments, literature searches, and past runs
   - Added clarification that MVP uses in-memory state + filesystem for intermediate results
   - Removed `cache/` directory from output structure

2. **Corrected Errors**
   - Fixed `lancedb-cpu` package name to `lancedb`
   - Fixed score range from `ge=0` to `ge=1` in Pydantic models (scores are 1-10, not 0-10)
   - Fixed CLI `choices` for score-threshold to use `range(1, 11)` instead of `range(0, 11)`
   - Fixed vector database to use LanceDB for both vectors AND metadata (removed unnecessary SQLite database)

3. **Clarified Unclear Information**
   - Added Section 3.4 "State Management" to clarify the MVP approach
   - Added embedding dimension explanation (100 tokens × 16 dim = 1600)
   - Clarified dataset statistics based on STACK paper
   - Added docstrings to all class `__init__` methods
   - Added results caching to "Future Extensions" (Section 16.1)

4. **Improved Consistency**
   - Aligned subagent definitions with DeepAgents patterns
   - Used consistent model string format (`provider:model_name`)
   - Standardized atlas names (LanceDB vs lancedb)