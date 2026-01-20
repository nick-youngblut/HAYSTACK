# Sprint 04: Agent Framework

**Duration**: 2-3 weeks  
**Dependencies**: Sprint 02 (Core Backend), Sprint 03 (Retrieval Strategies)  
**Goal**: Implement the orchestrator agent and subagents using LangChain/DeepAgents.

---

## Overview

> **Spec Reference**: `./specification/agents.md`, `./specification/tools.md`

This sprint implements the agentic AI system:
- Orchestrator agent with iteration control
- Query Understanding subagent
- Prompt Generation subagent
- Grounding Evaluation subagent
- LangChain tools for all operations

**Key Libraries** (per `./specification/dependencies.md`):
- `langchain>=1.0.0` - Core agent framework
- `deepagents>=0.1.0` - Agent building utilities
- `langgraph>=0.1.0` - Multi-agent orchestration
- `langchain-anthropic`, `langchain-openai`, `langchain-google-genai` - LLM providers

**Agent Initialization Pattern**:
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=0.7)
```

---

## Phase 1: LangChain Tool Definitions

> **Spec Reference**: `./specification/tools.md`

### Task 1.1: Database Query Tools

- [ ] **1.1.1** Create `orchestrator/tools/database_tools.py`:

```python
@tool
async def search_cells_by_perturbation(
    perturbation_name: str,
    cell_type_cl_id: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Search for cells by perturbation name with optional filters."""

@tool
async def search_cells_by_cell_type(
    cell_type_cl_id: str,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Search for cells by Cell Ontology ID."""

@tool
async def semantic_search_cells(
    query_text: str,
    search_type: Literal["perturbation", "cell_type", "sample_context"],
    top_k: int = 50,
    similarity_threshold: float = 0.7,
) -> str:
    """Vector similarity search for cells using text embeddings."""

@tool
async def find_donor_cells(
    donor_id: str,
    cell_type_cl_id: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Find cells for a specific donor."""
```

- [ ] **1.1.2** Format all outputs as YAML for LLM consumption
- [ ] **1.1.3** Add comprehensive docstrings

---

### Task 1.2: Cell Ontology Tools

> **Spec Reference**: `./specification/ontology-resolution.md` (Section 5)

- [ ] **1.2.1** Create `orchestrator/tools/ontology_tools.py`:

```python
@tool
async def resolve_cell_type_semantic(
    cell_labels: str,  # Semicolon-separated
    k: int = 3,
    distance_threshold: float = 0.7,
) -> str:
    """Map free-text cell type labels to Cell Ontology terms."""

@tool
async def get_cell_type_neighbors(
    term_ids: str,  # Semicolon-separated CL IDs
) -> str:
    """Get related Cell Ontology terms through relationships."""

@tool
async def query_cell_ontology_ols(
    search_terms: str,  # Semicolon-separated
) -> str:
    """Query OLS API as fallback for cell type resolution."""
```

---

### Task 1.3: Drug/Perturbation Tools

- [ ] **1.3.1** Create `orchestrator/tools/knowledge_tools.py`:

```python
@tool
async def resolve_perturbation(
    perturbation_name: str,
) -> str:
    """Resolve perturbation name to canonical form with external IDs."""

@tool
async def get_drug_targets(
    drug_name: str,
) -> str:
    """Get known target genes for a drug or compound."""

@tool
async def get_pathway_priors(
    perturbation_name: str,
) -> str:
    """Get expected pathway activations for a perturbation."""
```

---

### Task 1.4: Literature Tools

> **Spec Reference**: `./specification/literature-search.md`

- [ ] **1.4.1** Create `orchestrator/tools/literature_tools.py`:

```python
@tool
async def search_literature(
    query: str,
    max_results: int = 10,
    databases: Optional[str] = None,  # Semicolon-separated: pubmed, semantic_scholar, biorxiv
) -> str:
    """Search scientific literature databases for relevant papers."""

@tool
async def acquire_full_text_paper(
    doi: str,
) -> str:
    """Acquire full-text paper content and convert to markdown."""
```

---

### Task 1.5: Enrichment and Evaluation Tools

> **Spec Reference**: `./specification/tools.md` (Section 6.5)

- [ ] **1.5.1** Create `orchestrator/tools/enrichment_tools.py`:

```python
@tool
async def extract_de_genes(
    predictions_path: str,
    control_path: str,
    control_strategy: str,
    log2fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> str:
    """Extract differentially expressed genes from predictions."""

@tool
async def run_pathway_enrichment(
    gene_list: str,  # Semicolon-separated
    gene_sets: str = "GO_Biological_Process;KEGG_2021;Reactome_2022",
) -> str:
    """Run pathway enrichment analysis on a gene list."""

@tool
async def get_cell_type_markers(
    cell_type_cl_id: str,
) -> str:
    """Get canonical marker genes for a cell type."""

@tool
async def check_target_activation(
    de_genes: str,  # Semicolon-separated
    expected_targets: str,  # Semicolon-separated
) -> str:
    """Check if expected target genes are differentially expressed."""
```

---

## Phase 2: Query Understanding Subagent

> **Spec Reference**: `./specification/agents.md` (Section 5.2)

### Task 2.1: Implement QueryUnderstandingAgent

- [ ] **2.1.1** Create `orchestrator/agents/query_understanding.py`:

```python
class QueryUnderstandingAgent:
    """Parses natural language queries into structured form."""
    
    def __init__(self, model: BaseChatModel):
        self.agent = create_deep_agent(
            model=model,
            tools=[
                resolve_cell_type_semantic,
                get_cell_type_neighbors,
                query_cell_ontology_ols,
                resolve_perturbation,
                get_drug_targets,
                get_pathway_priors,
                search_literature,
            ],
            system_prompt=QUERY_UNDERSTANDING_PROMPT,
        )
    
    async def run(self, raw_query: str) -> StructuredQuery:
        """Parse query and resolve all entities."""
```

- [ ] **2.1.2** Create system prompt with:
  - Task type classification rules
  - Cell type resolution workflow (semantic → neighbors → OLS)
  - Perturbation resolution for perturbational tasks
  - Observational context extraction

---

### Task 2.2: Implement Cell Type Resolution Fallback

- [ ] **2.2.1** Create robust resolution chain:

```python
async def resolve_cell_type_with_fallback(label: str) -> dict:
    """Resolve cell type with graceful fallback through multiple methods."""
    # 1. Semantic search
    try:
        results = await ontology_service.semantic_search([label])
        if results.get(label) and results[label][0]["distance"] < 0.3:
            return {"resolved": True, "method": "semantic", ...}
    except Exception:
        pass
    
    # 2. OLS fallback
    try:
        ols_results = await ontology_service.query_ols([label])
        if ols_results.get(label):
            return {"resolved": True, "method": "ols", ...}
    except Exception:
        pass
    
    # 3. No resolution
    return {"resolved": False, "warning": "Could not resolve cell type to CL ID"}
```

---

## Phase 3: Prompt Generation Subagent

> **Spec Reference**: `./specification/agents.md` (Section 5.3)

### Task 3.1: Implement PromptGenerationAgent

- [ ] **3.1.1** Create `orchestrator/agents/prompt_generation.py`:

```python
class PromptGenerationAgent:
    """Generates prompt candidates for STACK inference."""
    
    def __init__(self, model: BaseChatModel, db: HaystackDatabase):
        self.agent = create_deep_agent(
            model=model,
            tools=[
                search_cells_by_perturbation,
                search_cells_by_cell_type,
                semantic_search_cells,
                find_donor_cells,
                get_cell_type_neighbors,
            ],
            system_prompt=PROMPT_GENERATION_PROMPT,
        )
        self.retriever = StrategyOrchestrator(db)
        self.ranker = CandidateRanker()
    
    async def run(
        self,
        query: StructuredQuery,
        previous_scores: list[int] = None,
    ) -> list[PromptCandidate]:
        """Generate and rank prompt candidates."""
```

- [ ] **3.1.2** Create system prompt with:
  - Strategy priority per task type
  - Ranking criteria
  - Control matching instructions

---

### Task 3.2: Integrate Retrieval Pipeline

- [ ] **3.2.1** Connect to retrieval strategies from Sprint 03
- [ ] **3.2.2** Apply candidate ranking
- [ ] **3.2.3** Match control prompts for synthetic control

---

## Phase 4: Grounding Evaluation Subagent

> **Spec Reference**: `./specification/agents.md` (Section 5.4)

### Task 4.1: Implement GroundingEvaluationAgent

- [ ] **4.1.1** Create `orchestrator/agents/grounding_evaluation.py`:

```python
class GroundingEvaluationAgent:
    """Evaluates biological grounding of predictions."""
    
    def __init__(self, model: BaseChatModel):
        self.agent = create_deep_agent(
            model=model,
            tools=[
                extract_de_genes,
                run_pathway_enrichment,
                check_target_activation,
                get_cell_type_markers,
                search_literature,
            ],
            system_prompt=GROUNDING_EVALUATION_PROMPT,
        )
    
    async def evaluate(
        self,
        predictions: AnnData,
        control_or_reference: AnnData,
        query: StructuredQuery,
        control_strategy: ControlStrategy,
    ) -> Union[GroundingScore, ObservationalGroundingScore]:
        """Evaluate predictions based on task type."""
```

---

### Task 4.2: Implement Perturbational Scoring

- [ ] **4.2.1** Score components (each 1-10):
  - **Pathway coherence**: Enriched pathways match expected biology
  - **Target activation**: Known targets differentially expressed
  - **Literature support**: Predictions have published evidence
  - **Network coherence**: DE genes form functional modules

- [ ] **4.2.2** Create evaluation prompt with scoring rubric

---

### Task 4.3: Implement Observational Scoring

- [ ] **4.3.1** Score components (each 1-10):
  - **Marker gene expression**: Canonical cell type markers expressed
  - **Tissue signature match**: Expression matches tissue-specific patterns
  - **Donor effect capture**: Donor-specific effects preserved
  - **Cell type coherence**: Expression consistent with cell identity

---

### Task 4.4: Control Strategy Confidence Adjustment

- [ ] **4.4.1** Apply confidence factors:
  - Synthetic control: Full confidence (paired comparison)
  - Query-as-control: -0.5 to -1.0 penalty (potential artifacts)

---

## Phase 5: Orchestrator Agent

> **Spec Reference**: `./specification/agents.md` (Section 5.1), `./specification/orchestrator.md`

### Task 5.1: Implement OrchestratorAgent

- [ ] **5.1.1** Create `orchestrator/agents/orchestrator.py`:

```python
class OrchestratorAgent:
    """Main orchestrator that runs the iterative HAYSTACK workflow."""
    
    def __init__(
        self,
        run_id: str,
        query: str,
        user_email: str,
        config: dict,
        control_strategy: ControlStrategy,
    ):
        self.run_id = run_id
        self.query = query
        self.config = config
        self.control_strategy = control_strategy
        self.max_iterations = config.get("max_iterations", 5)
        self.score_threshold = config.get("score_threshold", 7)
        
        # Initialize subagents
        self.query_agent = QueryUnderstandingAgent()
        self.prompt_agent = PromptGenerationAgent()
        self.evaluation_agent = GroundingEvaluationAgent()
    
    async def run(self) -> OrchestratorResult:
        """Execute the full iterative workflow."""
```

---

### Task 5.2: Implement Iteration Loop

- [ ] **5.2.1** Phase 1: Query Understanding
  - Parse query → StructuredQuery
  - Resolve entities (CL, DrugBank/PubChem)
  - Validate control strategy feasibility
  - Apply fallback if needed

- [ ] **5.2.2** Phase 2: Iteration Loop (max 5)
  - **Prompt Generation**: Run strategies, rank candidates
  - **Inference**: Submit GPU Batch job (next sprint)
  - **Evaluation**: Compute grounding score
  - Check convergence (score ≥ 7 or max iterations)
  - Check cancellation

- [ ] **5.2.3** Phase 3: Output Generation
  - Write final AnnData to GCS
  - Generate interpretation report
  - Write structured log

---

### Task 5.3: Implement Status Updates

- [ ] **5.3.1** Update database at each phase transition:

```python
async def _update_phase(self, phase: str, iteration: int = None):
    await database.update_run(
        run_id=self.run_id,
        current_phase=phase,
        current_iteration=iteration,
        updated_at=datetime.utcnow(),
    )
```

---

### Task 5.4: Implement Cancellation Handling

- [ ] **5.4.1** Check cancellation flag before each iteration:

```python
async def _is_cancelled(self) -> bool:
    run = await database.get_run(self.run_id)
    return run["status"] == "cancelled"
```

---

## Phase 6: Testing

### Task 6.1: Unit Tests

- [ ] **6.1.1** Test each tool with mocked services
- [ ] **6.1.2** Test subagents with mocked LLM responses
- [ ] **6.1.3** Test orchestrator with mock workflow

### Task 6.2: Integration Tests

- [ ] **6.2.1** Test full query understanding pipeline
- [ ] **6.2.2** Test full prompt generation pipeline
- [ ] **6.2.3** Test grounding evaluation with sample data

---

## Definition of Done

- [ ] All LangChain tools implemented with proper schemas
- [ ] Query Understanding agent resolves entities correctly
- [ ] Prompt Generation agent creates ranked candidates
- [ ] Grounding Evaluation agent scores predictions
- [ ] Orchestrator coordinates full workflow
- [ ] Cancellation handling works
- [ ] Unit tests pass (>80% coverage)

---

## Next Sprint

**Sprint 05: STACK Inference Integration** - Implement the GPU Batch job for STACK model inference.
