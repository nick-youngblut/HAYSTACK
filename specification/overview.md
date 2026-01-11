# HAYSTACK: Iterative Knowledge-Guided Cell Prompting System

**H**euristic **A**gent for **Y**ielding **S**TACK-**T**uned **A**ssessments with **C**losed-loop **K**nowledge

*Finding the optimal prompt in a haystack of possibilities*

---

## 1. Executive Summary

HAYSTACK is an agentic AI system that transforms STACK—a single-cell foundation model—from an open-loop inference tool into a closed-loop optimization system. By coupling multi-strategy prompt generation with biological grounding evaluation, HAYSTACK iteratively refines cell prompts until predictions align with pathway knowledge, literature evidence, and biological priors.

Given a natural language query, HAYSTACK autonomously:
1. Parses biological intent and resolves entities to standardized ontologies
2. Retrieves optimal prompt cells from ~10M cells across multiple atlases
3. Executes STACK in-context learning inference
4. Evaluates predictions against external biological knowledge
5. Iterates until convergence or stopping criteria are met

The result is biologically grounded predictions with interpretable confidence scores, downloadable gene expression data, and literature-cited reports.

---

## 2. Motivation: Why HAYSTACK?

### 2.1 The STACK Foundation Model

STACK (Single-cell Tabular Attention for Cell Knowledge) is a foundation model trained on 149 million human single cells that leverages **cellular context** to generate enhanced representations. Unlike prior single-cell foundation models that operate independently on each cell, STACK employs tabular attention across both gene and cell dimensions, enabling:

- **Zero-shot embedding**: Enhanced cell representations informed by context
- **In-context learning (ICL)**: Predicting cell states by engineering the cellular context
- **Cell prompting**: Using prompt cells to condition predictions on specific biological states

After post-training on 55M cells from CELLxGENE and the Parse PBMC dataset, STACK enables **cell prompting tasks** where prompt cells implicitly encode a biological condition (perturbation, disease state, donor variability) that guides predictions for query cells.

### 2.2 The Challenge: Manual Prompt Selection

Despite STACK's powerful capabilities, using it effectively requires:

1. **Domain expertise**: Knowing which cells to use as prompts for a given question
2. **Data familiarity**: Understanding what's available across heterogeneous atlases
3. **Manual iteration**: Trial-and-error to find prompts that yield biologically sensible results
4. **External validation**: Manually checking if predictions align with known biology

This creates a significant barrier for researchers who want to leverage STACK for novel biological questions without deep expertise in the training data or extensive manual experimentation.

### 2.3 The HAYSTACK Solution

HAYSTACK automates this entire workflow through an agentic system that:

| Challenge | HAYSTACK Solution |
|-----------|-------------------|
| "What cells should I use as prompts?" | Multi-strategy retrieval (mechanistic, semantic, ontological) automatically selects optimal prompts |
| "How do I know if predictions are valid?" | Biological grounding evaluation scores predictions against pathway databases and literature |
| "What if my first attempt isn't good?" | Iterative refinement automatically improves prompts based on evaluation feedback |
| "How do I interpret the results?" | LLM-generated reports explain predictions with literature citations |

---

## 3. Problems HAYSTACK Solves

### 3.1 Perturbation Effect Prediction

**Problem**: "I want to know how my cell type would respond to a treatment, but I don't have experimental data for that combination."

**Solution**: HAYSTACK finds cells that have been perturbed with similar treatments (or treatments with the same mechanism) and uses them as prompts to predict the treatment effect on your cell type of interest.

**Biological value**: Enables *in silico* perturbation screening before expensive wet lab experiments, prioritizing candidates for experimental validation.

### 3.2 Cross-Donor Cell Type Imputation

**Problem**: "My patient sample is missing certain cell types due to tissue availability or technical dropout. I need to impute what those cells would look like in this specific patient context."

**Solution**: HAYSTACK uses cells from other donors as templates (queries) and the patient's available cells as context (prompts) to generate patient-specific predictions for the missing cell types.

**Biological value**: Completes partial patient profiles for downstream analysis, enables cross-patient comparisons even with missing data.

### 3.3 Cross-Dataset Expression Generation

**Problem**: "I have perturbation data for some cell types but not others. I want to predict how a drug would affect cell types that were never experimentally perturbed."

**Solution**: HAYSTACK combines perturbation data from one source with cell type templates from independent atlases to generate predictions for novel combinations.

**Biological value**: Extends the utility of existing perturbation studies to cell types that are difficult to perturb experimentally, such as rare populations or tissue-resident cells.

### 3.4 Hypothesis Validation at Scale

**Problem**: "I have a hypothesis about how a pathway should respond to treatment, but I need to check it across multiple cell types and conditions."

**Solution**: HAYSTACK evaluates predictions against known biology (pathway enrichment, target activation, literature evidence) and provides confidence scores that indicate whether predictions align with expectations.

**Biological value**: Rapid hypothesis testing and validation before committing experimental resources.

---

## 4. Supported Task Types

HAYSTACK supports three categories of in-context learning tasks, mirroring STACK's capabilities:

### 4.1 Perturbational ICL

**Prompts**: Perturbed cells (drug-treated, cytokine-stimulated, genetically modified)  
**Queries**: Control cells (unperturbed)  
**Goal**: Predict how query cells would respond to the perturbation represented by prompts

| Task | Description | Example |
|------|-------------|---------|
| Novel cell types | Predict perturbation effect in cell types not in the prompt sample | Predict IL-6 effect on B cells using IL-6-treated T cells as prompts |
| Novel samples | Predict perturbation effect in a new donor/sample | Predict IFN-β response in a new patient using reference IFN-β data |

### 4.2 Observational ICL

**Prompts**: Cells from target donor/condition (the context you want to predict in)  
**Queries**: Cells from reference donors (providing cell type templates)  
**Goal**: Generate donor-specific or condition-specific cell type expression

| Task | Description | Example |
|------|-------------|---------|
| Cell type imputation | Predict expression of missing cell types in a donor | Impute fibroblasts for Patient A using Patient B's fibroblasts as template |
| Donor expression | Capture donor-specific variation | Generate kidney podocytes in a diabetic context |

### 4.3 Hybrid ICL

**Prompts**: Cells from one dataset/condition  
**Queries**: Cell types from a different dataset  
**Goal**: Generate cell types absent from the prompt dataset

| Task | Description | Example |
|------|-------------|---------|
| Cross-dataset generation | Generate cell types from one atlas in another's context | Predict drug effects on dendritic cells using drug-perturbed PBMC + independent DC atlas |

---

## 5. System Architecture Overview

HAYSTACK uses a **two-tier GCP Batch architecture** for robust, long-running agentic workflows:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Cloud Run)                              │
│   • FastAPI backend for job submission and status polling                       │
│   • Next.js frontend for query input and result visualization                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CPU BATCH JOB (Orchestrator Agent)                         │
│   • Query understanding and entity resolution                                   │
│   • Multi-strategy prompt generation                                            │
│   • Grounding evaluation with pathway/literature analysis                       │
│   • Iterative refinement until convergence                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      GPU BATCH JOB (STACK Inference)                            │
│   • A100 80GB for model inference                                               │
│   • Mask-diffusion generation procedure                                         │
│   • Returns predicted gene expression                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **Query Understanding Agent** | Parses natural language, resolves cell types (Cell Ontology), perturbations (DrugBank/PubChem), tissues (UBERON), diseases (MONDO) |
| **Prompt Generation Agent** | Executes parallel retrieval strategies, ranks candidates, selects optimal prompt cells |
| **Grounding Evaluation Agent** | Extracts DE genes, runs pathway enrichment, checks literature support, computes composite scores |
| **Cloud SQL (PostgreSQL + pgvector)** | Stores ~10M cell metadata records with text embeddings for semantic search |
| **GCS** | Atlas H5AD files, STACK model checkpoints, batch I/O, and output artifacts |

---

## 6. Example Workflows

### Example 1: Perturbation Effect Prediction (with Control Strategy Comparison)

**Query**: "How would lung fibroblasts respond to TGF-beta treatment?"

**User Configuration**:
- Control Strategy: `synthetic_control` (recommended)
- Max Iterations: 5
- Score Threshold: 7

---

#### Phase 1: Query Understanding

```yaml
task_type: PERTURBATION_NOVEL_CELL_TYPES
cell_type:
  query: "lung fibroblast"
  resolved_cl_id: CL:0002553
  resolved_name: "fibroblast of lung"
perturbation:
  query: "TGF-beta"
  resolved_type: CYTOKINE
  targets: [SMAD2, SMAD3, SMAD4, TGFBR1, TGFBR2]
  expected_pathways: ["TGF-beta signaling", "ECM organization", "EMT"]

control_strategy: SYNTHETIC_CONTROL
control_cells_available: true
control_cell_info:
  donor: "donor_1"
  condition: "PBS"
  cell_count: 156
  cell_types: ["T cell", "B cell", "Monocyte"]
```

---

#### Phase 2: Prompt Generation (with Paired Controls)

```yaml
selected_prompt:
  strategy: semantic_match
  description: "TGF-β1 treatment on Parse PBMC monocytes (fibroblast-like phenotype)"
  
  prompt_cell_indices: [45023, 45024, ..., 45150]  # 128 cells
  prompt_metadata:
    dataset: parse_pbmc
    donor: donor_1
    perturbation: "TGF-β1 (10 ng/mL, 24h)"
    cell_type: "CD14+ Monocyte"
  
  paired_control_indices: [12001, 12002, ..., 12156]  # 156 cells
  paired_control_metadata:
    dataset: parse_pbmc
    donor: donor_1
    perturbation: "PBS (vehicle control)"
    cell_type: "CD14+ Monocyte"

query_cell_indices: [...]  # 64 Tabula Sapiens lung fibroblasts
```

---

#### Phase 3: STACK Inference (Synthetic Control)

```yaml
perturbed_inference:
  prompt: TGF-β1 treated monocytes (donor_1)
  query: Lung fibroblasts (Tabula Sapiens)
  output: perturbed_predictions.h5ad
  duration: 3.2 minutes

control_inference:
  prompt: PBS treated monocytes (donor_1)
  query: Lung fibroblasts (Tabula Sapiens)
  output: control_predictions.h5ad
  duration: 3.1 minutes

total_inference_time: 6.3 minutes
```

---

#### Phase 4: Grounding Evaluation (Synthetic Control DE)

```yaml
differential_expression:
  control_strategy: SYNTHETIC_CONTROL
  treatment: perturbed_predictions.h5ad
  control: control_predictions.h5ad
  method: wilcoxon
  
  note: "DE computed between two STACK predictions, not raw query cells"

de_results:
  genes_up: 234
  genes_down: 89
  top_upregulated:
    - {gene: COL1A1, log2fc: 2.8, padj: 1.2e-15}
    - {gene: FN1, log2fc: 2.3, padj: 4.5e-12}
    - {gene: ACTA2, log2fc: 1.9, padj: 2.1e-10}

grounding_scores:
  pathway_coherence: 9/10
  target_activation: 8/10
  literature_support: 9/10
  network_coherence: 7/10
  
  composite_score: 8/10

confidence_note: "Synthetic control approach provides high-confidence DE results"
```

---

#### Alternative: Query-as-Control (for comparison)

If the user had selected `query_as_control`:

```yaml
differential_expression:
  control_strategy: QUERY_AS_CONTROL
  treatment: perturbed_predictions.h5ad
  control: query_cells.h5ad
  
  note: "DE computed between STACK prediction and original query cells"

de_results:
  genes_up: 312
  genes_down: 145
  
grounding_scores:
  composite_score: 7/10
  
confidence_note: |
  Query-as-control approach may include prompting artifacts.
  Consider re-running with synthetic control for publication-quality results.
```

---

### Example 2: Cross-Donor Cell Type Imputation

#### User Query
```
"Impute what macrophages would look like in kidney donor KD_047 who has chronic kidney disease"
```

#### Workflow Execution

**Phase 1: Query Understanding**
```yaml
Task Type: CELL_TYPE_IMPUTATION
Cell Type Resolution:
  Query: "macrophages"
  CL ID: CL:0000235 (macrophage)
  
Target Context:
  Donor ID: KD_047
  Tissue: Kidney (UBERON:0002113)
  Disease: Chronic kidney disease (MONDO:0005300)
  
Expected Markers:
  Pan-macrophage: CD68, CD14, FCGR3A
  Tissue-resident: LYVE1, FOLR2, MRC1
  Inflammatory: IL1B, TNF, CCL2
  
Literature Context:
  - CKD associated with pro-inflammatory macrophage phenotype
  - M1/M2 balance shifts toward inflammation in fibrotic kidney
```

**Phase 2: Prompt Generation**
```yaml
Strategy: donor_context + tissue_atlas
Prompt Cells (defining target donor context):
  - Dataset: Kidney Atlas
    Donor: KD_047 (CKD patient)
    Cell Types: Proximal tubule, Distal tubule, Endothelial
    N Cells: 96
    Rationale: "Other cell types from target donor define the disease context"
    
Query Cells (providing macrophage template):
  - Dataset: Kidney Atlas
    Donors: KD_012, KD_023, KD_031 (other CKD patients)
    Cell Type: Macrophage
    N Cells: 64
    Rationale: "Macrophages from similar disease context as templates"
```

**Phase 3: STACK Inference**
- Generates donor-specific macrophage expression
- Captures KD_047's specific disease signature

**Phase 4: Grounding Evaluation**
```yaml
Observational Grounding Scores:
  Marker Gene Expression: 9/10 (CD68, CD14, FCGR3A detected)
  Tissue Signature Match: 8/10 (kidney macrophage profile)
  Donor Effect Capture: 7/10 (CKD-associated genes elevated)
  Cell Type Coherence: 8/10 (clusters with reference macrophages)
  ─────────────────────
  Composite Score: 8/10 ✓
```

#### Workflow Results
```yaml
Interpretation Summary:
  "Generated macrophages for donor KD_047 show a pro-inflammatory 
   phenotype consistent with CKD, including elevated IL1B, CCL2, 
   and reduced FOLR2 compared to healthy kidney macrophages. 
   The cells express canonical macrophage markers (CD68+, CD14+) 
   and integrate well with the donor's other cell types."
```

---

### Example 3: Novel Drug Effect Prediction with Iteration

#### User Query
```
"What genes would be differentially expressed if I treated NK cells with ibrutinib?"
```

#### Workflow Execution

**Phase 1: Query Understanding**
```yaml
Task Type: PERTURBATION_NOVEL_CELL_TYPES
Cell Type: NK cell (CL:0000623)
Perturbation:
  Name: Ibrutinib
  Type: DRUG
  Mechanism: BTK inhibitor
  Targets: [BTK, ITK, EGFR, TEC]
  Expected Pathways:
    - B cell receptor signaling (actually less relevant for NK)
    - T cell receptor signaling (via ITK)
    - NK cell cytotoxicity pathway
```

**Phase 2-4: Iteration 1**
```yaml
Prompt Selection:
  Strategy: mechanistic_match
  Cells: BTK inhibitor-treated B cells (OpenProblems)
  Rationale: "Same drug class, different cell type"
  
Grounding Evaluation:
  Pathway Coherence: 4/10 (B cell pathways enriched - wrong context)
  Target Activation: 5/10 (BTK targets present but NK-specific missing)
  Literature Support: 3/10 (Limited evidence for this prompt choice)
  Composite Score: 4/10 ✗ (Below threshold)
  
Improvement Suggestion:
  "BTK is less critical in NK cells than ITK. Search for 
   ITK-related perturbations or T cell receptor modulators."
```

**Phase 2-4: Iteration 2**
```yaml
Prompt Selection (Refined):
  Strategy: mechanistic_match + semantic_match
  Cells: 
    - Dasatinib-treated T cells (targets ITK)
    - IL-2-stimulated NK cells (positive control for NK activation)
  Rationale: "ITK-targeting drug on related lymphocytes + NK-specific context"
  
Grounding Evaluation:
  Pathway Coherence: 7/10 (NK cytotoxicity pathways affected)
  Target Activation: 8/10 (ITK downstream genes appropriately modulated)
  Literature Support: 7/10 (Papers support ITK role in NK cell function)
  Composite Score: 7/10 ✓ (Meets threshold)
```

#### Workflow Results
```yaml
Run Summary:
  Iterations: 2
  Final Score: 7/10
  
Key Findings:
  - Ibrutinib predicted to moderately suppress NK cytotoxicity
  - Downregulation of GZMB, PRF1 (cytotoxic effectors)
  - Reduced IFNG production
  - Effect mediated primarily through ITK, not BTK
  
Clinical Relevance:
  "These predictions suggest ibrutinib treatment may partially 
   impair NK cell function, which could be relevant for 
   monitoring tumor immunosurveillance in CLL patients."
```

---

### Example 4: Hybrid Cross-Dataset Generation

#### User Query
```
"Generate what hepatocytes would look like if treated with the drugs from the OpenProblems PBMC dataset"
```

#### Workflow Execution

**Phase 1: Query Understanding**
```yaml
Task Type: CROSS_DATASET_GENERATION
Cell Type: Hepatocyte (CL:0000182)
Source Perturbations: OpenProblems drug screen (147 compounds)
Target Tissue: Liver (UBERON:0002107)
```

**Phase 2: Prompt Generation**
```yaml
Strategy: cross_dataset_hybrid

For each drug (parallelized):
  Prompt Cells:
    - Dataset: OpenProblems
      Cell Types: All PBMC types (B, T, NK, Monocyte)
      Perturbation: [drug_i]
      N Cells: 64
      
  Query Cells:
    - Dataset: Tabula Sapiens
      Tissue: Liver
      Cell Type: Hepatocyte
      Condition: Control
      N Cells: 32
```

**Phase 3-4: Batch Inference + Evaluation**
- Runs 147 inference jobs (batched for efficiency)
- Evaluates liver-specific responses
- Filters predictions by grounding scores

#### Workflow Results
```yaml
Output: Mini Perturb-Liver Atlas
  Compounds Processed: 147
  High-Confidence Predictions (score ≥ 7): 89
  Medium-Confidence (score 5-6): 41
  Low-Confidence (score < 5): 17
  
Top Hepatocyte-Relevant Predictions:
  1. Rifampicin: Strong CYP3A4 induction (expected, score 9/10)
  2. Dexamethasone: Glucocorticoid response (score 8/10)
  3. Acetaminophen: Oxidative stress signature (score 8/10)
  
Surprising Findings:
  - Several kinase inhibitors show hepatocyte-specific effects
    not predicted from PBMC responses alone
```

---

## 7. Output Artifacts

### 7.1 Files Generated

| File | Format | Description |
|------|--------|-------------|
| `{run_id}_predictions.h5ad` | AnnData | Predicted gene expression with full metadata |
| `{run_id}_report.html` | HTML | Interactive interpretation report with visualizations |
| `{run_id}_log.json` | JSON | Complete execution trace for reproducibility |

### 7.2 AnnData Structure

```python
predictions.obs:
    cell_id, task_type, cell_type_query, perturbation_query,
    target_donor_id, prompt_strategy, iteration, grounding_score
    
predictions.var:
    gene_symbol, predicted_lfc, predicted_pval, is_de, target_gene
    
predictions.uns:
    run_id, query, config, iterations, enrichment_results
```

### 7.3 Report Contents

The HTML report includes:
- **Executive summary**: Key findings in plain language
- **Differential expression table**: Sortable, filterable gene list
- **Pathway enrichment plots**: Interactive bar charts and dot plots
- **Literature citations**: DOI-linked references supporting predictions
- **Methodology notes**: Prompts used, iterations, scores per round

---

## 8. MVP Goals (Technical)

- Accept natural language ICL requests via a web interface
- Generate biologically informed prompts using multiple retrieval strategies
- Run STACK inference and evaluate predictions against biological knowledge
- Iterate until convergence with configurable thresholds (default: score ≥ 7, max 5 iterations)
- Provide polling status, cancellation, and downloadable results
- Send email notification on completion

---

## 9. Specification Map

| Document | Contents |
|----------|----------|
| `specification/architecture.md` | System design, component responsibilities, data flow |
| `specification/data-models.md` | Pydantic schemas for queries, candidates, scores |
| `specification/agents.md` | Orchestrator and subagent specifications |
| `specification/tools.md` | LangChain tool definitions |
| `specification/prompt-retrieval.md` | Cell retrieval strategies and ranking |
| `specification/ontology-resolution.md` | Cell Ontology integration |
| `specification/literature-search.md` | PubMed, Semantic Scholar, bioRxiv integration |
| `specification/database.md` | PostgreSQL + pgvector schema |
| `specification/backend-api.md` | FastAPI routes and models |
| `specification/orchestrator.md` | CPU Batch job implementation |
| `specification/frontend.md` | Next.js UI components |
| `specification/configuration.md` | Dynaconf settings |
| `specification/deployment.md` | GCP infrastructure |
| `specification/output.md` | Result file formats |
| `specification/error-handling.md` | Error taxonomy and recovery |
| `specification/testing.md` | Test strategy and fixtures |
| `specification/dependencies.md` | Python packages |
| `specification/future-extensions.md` | Post-MVP roadmap |
| `specification/appendices.md` | Grounding score formulas, strategy hierarchy |

---

## 10. References

- **STACK Paper**: Dong, M. et al. (2026). Stack: In-Context Learning of Single-Cell Biology. bioRxiv. https://doi.org/10.64898/2026.01.09.698608
- **Parse PBMC Dataset**: Parse Biosciences (2023). 10M PBMC cytokine perturbation atlas.
- **OpenProblems**: Luecken et al. (2025). Single-cell perturbation benchmark.
- **Tabula Sapiens**: Consortium et al. (2022). Multi-organ single-cell atlas.
- **Cell Ontology**: Diehl et al. (2016). CL: A reference ontology for cell types.
