# Sprint 12: Documentation & Launch Prep

**Duration**: 1 week  
**Dependencies**: All previous sprints  
**Goal**: Complete documentation and prepare for production launch.

---

## Overview

> **Spec Reference**: `./specification/README.md`, `./specification/dependencies.md`

This sprint implements:
- API documentation (OpenAPI)
- User guide
- Developer guide
- Data validation
- Performance optimization
- Security review
- Launch checklist

---

## Phase 1: API Documentation

### Task 1.1: Generate OpenAPI Spec

- [ ] **1.1.1** Verify FastAPI auto-generates OpenAPI spec at `/openapi.json`
- [ ] **1.1.2** Add comprehensive descriptions to all endpoints
- [ ] **1.1.3** Add request/response examples
- [ ] **1.1.4** Document error responses

---

### Task 1.2: Create Swagger UI Customization

- [ ] **1.2.1** Configure Swagger UI at `/docs`
- [ ] **1.2.2** Add HAYSTACK branding
- [ ] **1.2.3** Organize endpoints by tag

---

## Phase 2: User Guide

### Task 2.1: Create Query Formulation Guide

- [ ] **2.1.1** Write `docs/user-guide/queries.md`:

```markdown
# Formulating HAYSTACK Queries

## Perturbational Queries

Predict how cells would respond to perturbations:

**Good examples:**
- "How would lung fibroblasts respond to TGF-beta treatment?"
- "Predict the effect of ibrutinib on NK cells"
- "What genes would change in T cells after IL-6 exposure?"

**Required elements:**
1. Cell type (required): fibroblasts, T cells, macrophages
2. Perturbation (required): drug name, cytokine, genetic perturbation
3. Context (optional): tissue, disease state

## Observational Queries

Impute or predict cell type expression:

**Good examples:**
- "Impute macrophages for donor KD_047 with chronic kidney disease"
- "Predict fibroblast expression in a healthy lung sample"

**Required elements:**
1. Cell type (required)
2. Donor or tissue context (required)
```

---

### Task 2.2: Create Control Strategy Guide

- [ ] **2.2.1** Write `docs/user-guide/control-strategies.md`:

```markdown
# Control Strategies

## Synthetic Control (Recommended)

Uses matched unperturbed cells as control for differential expression.

**Pros:**
- Higher confidence in DE results
- Cleaner separation of perturbation effects
- Better for benchmarking

**Cons:**
- Requires matched control cells in atlas
- 2x inference time
- May not be available for all queries

## Query as Control

Uses original query cells as baseline for DE analysis.

**When to use:**
- When matched controls unavailable
- For exploratory analysis
- When speed is priority
```

---

### Task 2.3: Create Results Interpretation Guide

- [ ] **2.3.1** Write `docs/user-guide/results.md`:
  - Grounding score interpretation
  - DE gene lists
  - Pathway enrichment results
  - Literature citations

---

## Phase 3: Developer Guide

### Task 3.1: Create Setup Guide

- [ ] **3.1.1** Write `docs/developer/setup.md`:
  - Prerequisites
  - Local development setup
  - Docker Compose usage
  - Environment variables

---

### Task 3.2: Create Extension Guides

- [ ] **3.2.1** Write `docs/developer/adding-strategies.md`:
  - How to add new retrieval strategies
  - Strategy interface

- [ ] **3.2.2** Write `docs/developer/extending-evaluation.md`:
  - How to add new grounding metrics
  - Custom scoring functions

---

### Task 3.3: Create Deployment Guide

- [ ] **3.3.1** Write `docs/developer/deployment.md`:
  - GCP setup
  - Cloud Build
  - Secrets management
  - Monitoring

---

## Phase 4: Data Validation

### Task 4.1: Validate Atlas Data Quality

- [ ] **4.1.1** Verify cell counts:
  - Parse PBMC: ~10M cells
  - OpenProblems: ~500K cells
  - Tabula Sapiens: ~500K cells

- [ ] **4.1.2** Run validation queries:

```sql
-- Cell count by dataset
SELECT dataset, COUNT(*) as n_cells
FROM cells
GROUP BY dataset;

-- CL ID coverage
SELECT 
  dataset,
  COUNT(*) as total,
  COUNT(cell_type_cl_id) as with_cl_id,
  COUNT(cell_type_cl_id) * 100.0 / COUNT(*) as coverage_pct
FROM cells
GROUP BY dataset;
```

- [ ] **4.1.3** Document data quality metrics

---

### Task 4.2: Validate Index Completeness

- [ ] **4.2.1** Verify all cells have embeddings where expected
- [ ] **4.2.2** Verify HNSW indexes are used (EXPLAIN ANALYZE)
- [ ] **4.2.3** Test vector search performance

---

## Phase 5: Performance Optimization

### Task 5.1: Optimize Database Queries

- [ ] **5.1.1** Review slow query logs
- [ ] **5.1.2** Add missing indexes if needed
- [ ] **5.1.3** Tune HNSW parameters
- [ ] **5.1.4** Optimize connection pool settings

---

### Task 5.2: Optimize LLM Calls

- [ ] **5.2.1** Review token usage
- [ ] **5.2.2** Implement caching for repeated queries
- [ ] **5.2.3** Tune temperature and max_tokens

---

### Task 5.3: Optimize STACK Inference

- [ ] **5.3.1** Tune batch size for GPU memory
- [ ] **5.3.2** Profile inference time
- [ ] **5.3.3** Document optimal settings

---

## Phase 6: Security Review

### Task 6.1: Review Authentication

- [ ] **6.1.1** Verify IAP configuration
- [ ] **6.1.2** Test unauthorized access is blocked
- [ ] **6.1.3** Verify user email extraction

---

### Task 6.2: Review Permissions

- [ ] **6.2.1** Audit service account permissions
- [ ] **6.2.2** Verify least privilege principle
- [ ] **6.2.3** Document permission requirements

---

### Task 6.3: Review Secrets

- [ ] **6.3.1** Verify no secrets in code
- [ ] **6.3.2** Verify Secret Manager access controls
- [ ] **6.3.3** Rotate any exposed credentials

---

## Phase 7: Launch Checklist

### Task 7.1: Final Verification

- [ ] **7.1.1** All endpoints respond correctly
- [ ] **7.1.2** Run creation works end-to-end
- [ ] **7.1.3** Email notifications sent
- [ ] **7.1.4** Results downloadable
- [ ] **7.1.5** Monitoring dashboards functional
- [ ] **7.1.6** Alerts configured

---

### Task 7.2: Cost Estimation

| Control Strategy | 1 Iteration | 5 Iterations |
|-----------------|-------------|--------------|
| Query-as-Control | ~$0.30 | ~$1.25 |
| Synthetic Control | ~$0.55 | ~$2.25 |

**Monthly cost estimate** (100 runs/month):
- Compute: ~$200
- Cloud SQL: ~$300
- Storage: ~$10
- LLM API: ~$50
- **Total**: ~$560/month

---

### Task 7.3: Backup Verification

- [ ] **7.3.1** Test database backup restore
- [ ] **7.3.2** Verify GCS versioning works
- [ ] **7.3.3** Document disaster recovery procedure

---

## Definition of Done

- [ ] API documentation complete
- [ ] User guide complete
- [ ] Developer guide complete
- [ ] Data quality validated
- [ ] Performance optimized
- [ ] Security review passed
- [ ] All launch checklist items verified
- [ ] Cost estimates documented

---

## Post-Launch

- Monitor error rates
- Gather user feedback
- Plan next iteration of features
