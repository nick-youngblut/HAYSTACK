# Sprint 03: Prompt Retrieval Strategies

**Duration**: 2 weeks  
**Dependencies**: Sprint 02 (Core Backend Services)  
**Goal**: Implement the retrieval strategy framework and all six prompt retrieval strategies.

---

## Overview

> **Spec Reference**: `./specification/prompt-retrieval.md`

This sprint implements the prompt retrieval system:
- Base strategy framework with task-specific orchestration
- Six retrieval strategies (Direct, Mechanistic, Semantic, Ontology, Donor, Tissue)
- Candidate ranking and selection with diversity scoring
- Control strategy matching for synthetic control

---

## Phase 1: Strategy Framework

### Task 1.1: Implement Base RetrievalStrategy

- [ ] **1.1.1** Create `orchestrator/retrieval/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Optional

class RetrievalStrategy(ABC):
    """Base class for all retrieval strategies."""
    
    def __init__(self, db: HaystackDatabase):
        self.db = db
    
    @abstractmethod
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellSetCandidate]:
        """Retrieve candidate cell sets for the query."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name for logging."""
        pass
```

---

### Task 1.2: Implement Strategy Orchestrator

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 4: Strategy Priority)

- [ ] **1.2.1** Create task-specific priority lists:

```python
# Perturbational tasks: predict drug/cytokine effects
PERTURBATIONAL_STRATEGY_PRIORITY = [
    DirectMatchStrategy,      # Exact perturbation + cell type
    MechanisticMatchStrategy, # Same target genes/pathways
    SemanticMatchStrategy,    # Similar perturbation description
    OntologyGuidedStrategy,   # Related cell types via CL
]

# Observational tasks: impute/predict cell types for donors
OBSERVATIONAL_STRATEGY_PRIORITY = [
    DonorContextStrategy,     # Same donor, different cell types
    TissueAtlasStrategy,      # Same tissue, different donors
    OntologyGuidedStrategy,   # Related cell types via CL
    SemanticMatchStrategy,    # Similar sample context
]
```

- [ ] **1.2.2** Implement `execute_strategy_pipeline()`:
  - Determine task type from query
  - Execute strategies in priority order
  - Deduplicate using `selection_key()`
  - Stop when max_results reached

---

## Phase 2: Perturbational Strategies

### Task 2.1: Implement DirectMatchStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 5)

- [ ] **2.1.1** Create `orchestrator/retrieval/direct_match.py`:
  - Exact perturbation name + cell type matching
  - Include synonym lookup via synonyms table
  - Return highest quality cells (by n_genes, total_counts)

- [ ] **2.1.2** SQL query:

```sql
SELECT c.dataset, c.perturbation_name, c.cell_type_name, c.cell_type_cl_id,
       c.donor_id, c.tissue_uberon_id, c.sample_condition,
       ARRAY_AGG(c.cell_index) AS cell_indices,
       COUNT(*) AS n_cells,
       AVG(c.n_genes) AS mean_n_genes
FROM cells c
LEFT JOIN synonyms s ON LOWER(s.synonym) = LOWER($1) AND s.entity_type = 'perturbation'
WHERE (LOWER(c.perturbation_name) = LOWER($1) OR c.perturbation_name = s.canonical_name)
  AND c.cell_type_cl_id = $2
  AND c.is_control = FALSE
GROUP BY c.dataset, c.perturbation_name, c.cell_type_name, c.cell_type_cl_id,
         c.donor_id, c.tissue_uberon_id, c.sample_condition
ORDER BY n_cells DESC, mean_n_genes DESC
LIMIT $3
```

---

### Task 2.2: Implement MechanisticMatchStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 6)

- [ ] **2.2.1** Create `orchestrator/retrieval/mechanistic_match.py`:
  - Input: target genes and pathways from StructuredQuery
  - Find perturbations sharing targets/pathways
  - Score by Jaccard similarity

- [ ] **2.2.2** Implement target overlap scoring:

```python
async def _compute_mechanistic_score(
    self,
    query_targets: set[str],
    query_pathways: set[str],
    candidate_targets: set[str],
    candidate_pathways: set[str],
) -> float:
    """Compute mechanistic similarity via Jaccard."""
    target_overlap = len(query_targets & candidate_targets) / len(query_targets | candidate_targets) if query_targets else 0
    pathway_overlap = len(query_pathways & candidate_pathways) / len(query_pathways | candidate_pathways) if query_pathways else 0
    return 0.6 * target_overlap + 0.4 * pathway_overlap
```

---

### Task 2.3: Implement SemanticMatchStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 7)

- [ ] **2.3.1** Create `orchestrator/retrieval/semantic_match.py`:
  - Vector similarity search on perturbation_embedding
  - Configurable similarity threshold (default: 0.7)
  - Uses HNSW indexes

- [ ] **2.3.2** Generate query embedding from structured query:

```python
def _build_query_text(self, query: StructuredQuery) -> str:
    text = f"Perturbation: {query.perturbation_query}"
    if query.perturbation_type:
        text += f" ({query.perturbation_type.value})"
    if query.expected_targets:
        text += f" targeting {', '.join(query.expected_targets[:5])}"
    if query.expected_pathways:
        text += f" affecting {', '.join(query.expected_pathways[:3])} pathways"
    return text
```

---

### Task 2.4: Implement OntologyGuidedStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 8), `./specification/ontology-resolution.md`

- [ ] **2.4.1** Create `orchestrator/retrieval/ontology_guided.py`:
  - Uses native CellOntologyService (no external MCP)
  - CL hierarchy traversal for related cell types
  - Priority: parents > children > siblings
  - Score by ontology distance: `1.0 / (distance + 1)`

- [ ] **2.4.2** Implement neighbor retrieval:

```python
async def retrieve(self, query: StructuredQuery, max_results: int = 50, filters: dict = None):
    if not query.cell_type_cl_id:
        return []
    
    neighbors = await self.ontology_service.get_neighbors(
        term_ids=[query.cell_type_cl_id],
        relationship_types=["is_a", "part_of", "develops_from"],
        max_distance=2,
    )
    
    candidates = []
    for neighbor in neighbors:
        cells = await self._search_cells_for_type(
            cl_id=neighbor["term_id"],
            perturbation=query.perturbation_resolved,
        )
        for cell_set in cells:
            cell_set.ontology_distance = neighbor["distance"]
            cell_set.relevance_score = 1.0 / (neighbor["distance"] + 1)
            candidates.append(cell_set)
    
    return sorted(candidates, key=lambda x: -x.relevance_score)[:max_results]
```

---

## Phase 3: Observational Strategies

### Task 3.1: Implement DonorContextStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 9)

- [ ] **3.1.1** Create `orchestrator/retrieval/donor_context.py`:
  - For observational tasks (donor imputation, expression prediction)
  - Find same donor, different cell types for context
  - Prioritize unperturbed/control cells

- [ ] **3.1.2** SQL query:

```sql
SELECT c.dataset, c.cell_type_name, c.cell_type_cl_id,
       c.donor_id, c.tissue_name, c.tissue_uberon_id,
       c.sample_condition, c.sample_metadata,
       ARRAY_AGG(c.cell_index) AS cell_indices,
       COUNT(*) AS n_cells, AVG(c.n_genes) AS mean_n_genes
FROM cells c
WHERE c.donor_id = $1
  AND c.cell_type_cl_id != $2
  AND c.is_control = TRUE
GROUP BY c.dataset, c.cell_type_name, c.cell_type_cl_id,
         c.donor_id, c.tissue_name, c.tissue_uberon_id,
         c.sample_condition, c.sample_metadata
ORDER BY n_cells DESC
LIMIT $3
```

---

### Task 3.2: Implement TissueAtlasStrategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 10)

- [ ] **3.2.1** Create `orchestrator/retrieval/tissue_atlas.py`:
  - For cross-tissue cell type retrieval
  - Prioritizes comprehensive atlases (Tabula Sapiens)
  - Selects high-quality reference populations

- [ ] **3.2.2** SQL query with atlas prioritization:

```sql
SELECT c.dataset, c.cell_type_name, c.cell_type_cl_id,
       c.donor_id, c.tissue_name, c.tissue_uberon_id,
       ARRAY_AGG(c.cell_index) AS cell_indices,
       COUNT(*) AS n_cells
FROM cells c
WHERE c.cell_type_cl_id = $1
  AND c.is_control = TRUE
  AND ($2::text IS NULL OR c.tissue_uberon_id = $2)
GROUP BY c.dataset, c.cell_type_name, c.cell_type_cl_id,
         c.donor_id, c.tissue_name, c.tissue_uberon_id
ORDER BY 
    CASE c.dataset 
        WHEN 'tabula_sapiens' THEN 1 
        ELSE 2 
    END,
    n_cells DESC
LIMIT $3
```

---

## Phase 4: Candidate Ranking & Selection

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 11)

### Task 4.1: Implement CandidateRanker

- [ ] **4.1.1** Create `orchestrator/retrieval/ranker.py`:

```python
class CandidateRanker:
    """Rank and select final prompt candidates."""
    
    def __init__(
        self,
        relevance_weight: float = 0.4,  # Per spec
        quality_weight: float = 0.3,    # Per spec
        diversity_weight: float = 0.3,  # Per spec
    ):
        self.relevance_weight = relevance_weight
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
    
    def rank_candidates(
        self,
        candidates: list[CellSetCandidate],
        top_k: int = 10,
    ) -> list[CellSetCandidate]:
        """Rank candidates using greedy selection with diversity."""
        selected = []
        remaining = list(candidates)
        
        while remaining and len(selected) < top_k:
            best_score, best_idx = -1, -1
            for i, c in enumerate(remaining):
                score = self._compute_final_score(c, selected)
                if score > best_score:
                    best_score, best_idx = score, i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected
```

- [ ] **4.1.2** Implement quality scoring:

```python
def _compute_quality_score(self, candidate: CellSetCandidate) -> float:
    """Score based on data quality metrics."""
    score = 0.0
    
    # Cell count factor (log scale, cap at 1000)
    if candidate.n_cells:
        score += 0.4 * min(1.0, log10(candidate.n_cells + 1) / 3)
    
    # Gene count factor
    if candidate.mean_n_genes:
        score += 0.3 * min(1.0, candidate.mean_n_genes / 5000)
    
    # Dataset preference
    dataset_scores = {"tabula_sapiens": 0.3, "parse_pbmc": 0.25, "openproblems": 0.2}
    score += dataset_scores.get(candidate.dataset, 0.1)
    
    return score
```

- [ ] **4.1.3** Implement diversity scoring:

```python
def _compute_diversity_score(self, candidate: CellSetCandidate, selected: list) -> float:
    """Penalize redundant selections."""
    if not selected:
        return 1.0
    
    same_pert = sum(1 for s in selected if s.perturbation_name == candidate.perturbation_name)
    same_ct = sum(1 for s in selected if s.cell_type_cl_id == candidate.cell_type_cl_id)
    same_dataset = sum(1 for s in selected if s.dataset == candidate.dataset)
    
    diversity = 1.0
    diversity -= 0.3 * (same_pert / len(selected))
    diversity -= 0.2 * (same_ct / len(selected))
    diversity -= 0.1 * (same_dataset / len(selected))
    
    return max(0.0, diversity)
```

---

## Phase 5: Control Strategy Matching

> **Spec Reference**: `./specification/data-models.md` (ControlStrategy)

### Task 5.1: Implement Synthetic Control Matching

- [ ] **5.1.1** Create `orchestrator/retrieval/control_matching.py`:

```python
async def find_matched_controls(
    query: StructuredQuery,
    prompt_candidates: list[PromptCandidate],
    db: HaystackDatabase,
) -> list[PromptCandidate]:
    """Find matched control cells for each prompt candidate."""
    for candidate in prompt_candidates:
        # Find control cells from same donor/sample
        control_cells = await db.execute_query('''
            SELECT cell_index 
            FROM cells 
            WHERE donor_id = $1 
              AND cell_type_cl_id = $2
              AND is_control = TRUE
              AND sample_condition = $3
            LIMIT 128
        ''', (candidate.cell_set.donor_id, 
              candidate.cell_set.cell_type_cl_id,
              candidate.cell_set.sample_condition))
        
        if control_cells:
            candidate.paired_control_indices = [c["cell_index"] for c in control_cells]
            candidate.paired_control_metadata = {
                "donor_id": candidate.cell_set.donor_id,
                "cell_type": candidate.cell_set.cell_type_name,
                "n_cells": len(control_cells),
            }
    
    return prompt_candidates
```

- [ ] **5.1.2** Implement control availability check for API

---

### Task 5.2: Implement Control Strategy Fallback

> **Spec Reference**: `./specification/data-models.md` (ControlStrategy), `./INIT-IMP-PLAN.md` (Control Strategy section)

- [ ] **5.2.1** Implement fallback logic:

```python
async def determine_effective_control_strategy(
    requested_strategy: ControlStrategy,
    prompt_candidates: list[PromptCandidate],
) -> tuple[ControlStrategy, str]:
    """
    Determine effective control strategy based on availability.
    
    Fallback Order:
    1. If synthetic_control requested and controls found → synthetic_control
    2. If synthetic_control requested but no controls → query_as_control (with warning)
    3. If query_as_control requested → query_as_control
    """
    if requested_strategy == ControlStrategy.QUERY_AS_CONTROL:
        return ControlStrategy.QUERY_AS_CONTROL, "User requested query-as-control"
    
    # Check if any candidate has paired controls
    has_controls = any(c.paired_control_indices for c in prompt_candidates)
    
    if has_controls:
        return ControlStrategy.SYNTHETIC_CONTROL, "Matched controls found"
    else:
        return ControlStrategy.QUERY_AS_CONTROL, "Fallback: no matched controls available"
```

- [ ] **5.2.2** Document query-as-control workflow:
  - Single STACK inference (no paired control)
  - DE computed against original query cells
  - Store `query_cells_gcs_path` in IterationRecord
  - Apply mild confidence penalty in grounding scoring

- [ ] **5.2.3** Track control strategy in run record:
  - `control_strategy`: User's requested strategy
  - `control_strategy_effective`: Actual strategy used (after fallback)
  - `control_cells_available`: Boolean indicating if matched controls were found

---

## Phase 6: Testing & Validation

### Task 6.1: Unit Tests for Strategies

- [ ] **6.1.1** Test each strategy with mock database
- [ ] **6.1.2** Test ranking algorithm correctness
- [ ] **6.1.3** Test diversity scoring
- [ ] **6.1.4** Test control matching

### Task 6.2: Integration Tests

- [ ] **6.2.1** Test full retrieval pipeline with real database
- [ ] **6.2.2** Benchmark query performance (<1s for typical queries)
- [ ] **6.2.3** Test with various query types (perturbational, observational)

---

## Definition of Done

- [ ] All six retrieval strategies implemented
- [ ] Strategy orchestrator selects appropriate priority per task type
- [ ] Candidate ranker uses correct weights (0.4/0.3/0.3)
- [ ] Control matching finds paired controls when available
- [ ] Unit tests pass (>80% coverage)
- [ ] Performance: <1s for typical retrieval queries

---

## Next Sprint

**Sprint 04: Agent Framework** - Implement the orchestrator and subagents using LangChain/DeepAgents.
