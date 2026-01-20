# Sprint 01: Foundation & Infrastructure

**Duration**: 2-3 weeks  
**Dependencies**: None  
**Goal**: Establish the GCP infrastructure, database schema, and initial data pipelines.

---

## Overview

This sprint establishes the foundational infrastructure for HAYSTACK, including:
- GCP Cloud SQL with PostgreSQL 15 + pgvector
- GCS bucket configuration
- Database schema creation with vector support
- Atlas data harmonization and loading
- Cell Ontology integration

---

## Phase 1: GCP Infrastructure Setup

> **Spec Reference**: `./specification/database.md` (Section 8.1: Cloud SQL Setup)

### Task 1.1: Create GCP Project and Enable APIs

**Description**: Set up the GCP project and enable required APIs.

- [ ] **1.1.1** Create or select GCP project (`haystack-prod`)
- [ ] **1.1.2** Enable required APIs:
  - Cloud SQL Admin API
  - Cloud Storage API
  - Cloud Batch API
  - Cloud Run API
  - Secret Manager API
  - Identity-Aware Proxy API
  - Artifact Registry API
- [ ] **1.1.3** Configure billing and set up budget alerts
- [ ] **1.1.4** Document project configuration in README

**Acceptance Criteria**:
- All APIs enabled and accessible
- Billing configured with alerts at $100, $500, $1000

---

### Task 1.2: Configure VPC Network

**Description**: Set up VPC networking for secure Cloud SQL access.

> **Spec Reference**: `./specification/deployment.md` (Section 12.6: VPC Network)

- [ ] **1.2.1** Create VPC network (`haystack-vpc`)
- [ ] **1.2.2** Create Serverless VPC Access connector for Cloud Run
  ```bash
  gcloud compute networks vpc-access connectors create haystack-connector \
    --region=us-east1 \
    --network=haystack-vpc \
    --range=10.8.0.0/28
  ```
- [ ] **1.2.3** Configure firewall rules for internal communication
- [ ] **1.2.4** Allocate private IP range for Cloud SQL
- [ ] **1.2.5** Document network topology

**Acceptance Criteria**:
- VPC connector created and functional
- Cloud SQL private IP accessible from Cloud Run

---

### Task 1.3: Provision Cloud SQL Instance

**Description**: Create the PostgreSQL 15 + pgvector instance.

> **Spec Reference**: `./specification/database.md` (Section 8.1: Cloud SQL Setup)

- [ ] **1.3.1** Create Cloud SQL instance:
  ```bash
  gcloud sql instances create haystack-db \
    --database-version=POSTGRES_15 \
    --tier=db-custom-4-15360 \
    --region=us-east1 \
    --storage-size=100GB \
    --storage-type=SSD \
    --storage-auto-increase \
    --enable-point-in-time-recovery \
    --network=haystack-vpc \
    --no-assign-ip
  ```
- [ ] **1.3.2** Enable high availability (for production)
- [ ] **1.3.3** Configure maintenance window (Sunday 3 AM)
- [ ] **1.3.4** Create database user for application
- [ ] **1.3.5** Store credentials in Secret Manager
- [ ] **1.3.6** Enable pgvector extension
- [ ] **1.3.7** Test connectivity from local environment

**Acceptance Criteria**:
- Cloud SQL instance running with pgvector enabled
- Private IP accessible via VPC connector
- Credentials stored securely in Secret Manager

---

### Task 1.4: Create GCS Buckets

**Description**: Create and configure GCS buckets for data storage.

> **Spec Reference**: `./specification/architecture.md` (Section 3.3: Data Layer)

- [ ] **1.4.1** Create buckets with appropriate naming:
  - `haystack-atlases` - Atlas H5AD files (regional, standard)
  - `haystack-models` - STACK model checkpoints (regional, standard)
  - `haystack-batch-io` - Batch job I/O (regional, standard)
  - `haystack-results` - Output artifacts (regional, standard)
- [ ] **1.4.2** Configure lifecycle policies:
  - `haystack-batch-io`: Delete after 7 days
  - `haystack-results`: Delete after 90 days
- [ ] **1.4.3** Enable versioning on `haystack-models`
- [ ] **1.4.4** Set up appropriate IAM permissions
- [ ] **1.4.5** Create folder structure in each bucket

**Bucket Structure**:
```
haystack-atlases/
├── parse_pbmc/
│   └── parse_pbmc.h5ad
├── openproblems/
│   └── openproblems.h5ad
└── tabula_sapiens/
    └── tabula_sapiens.h5ad

haystack-models/
├── stack_v1/
│   ├── model.pt
│   └── genelist.txt

haystack-batch-io/
└── runs/
    └── {run_id}/
        ├── prompt.h5ad
        ├── query.h5ad
        └── predictions.h5ad

haystack-results/
└── runs/
    └── {run_id}/
        ├── output.h5ad
        ├── report.md
        └── log.json
```

**Acceptance Criteria**:
- All buckets created with correct settings
- Lifecycle policies applied
- IAM permissions configured

---

### Task 1.5: Configure Service Accounts

**Description**: Create service accounts with appropriate permissions.

> **Spec Reference**: `./specification/deployment.md` (Section 12.7: Service Accounts)

- [ ] **1.5.1** Create service accounts:
  - `haystack-cloudrun@` - For Cloud Run API
  - `haystack-orchestrator@` - For CPU Batch jobs
  - `haystack-inference@` - For GPU Batch jobs
- [ ] **1.5.2** Assign roles:
  
  **haystack-cloudrun@**:
  - `roles/cloudsql.client`
  - `roles/storage.objectAdmin` (for results bucket)
  - `roles/batch.jobsEditor`
  - `roles/secretmanager.secretAccessor`
  
  **haystack-orchestrator@**:
  - `roles/cloudsql.client`
  - `roles/storage.objectAdmin`
  - `roles/batch.jobsEditor`
  - `roles/secretmanager.secretAccessor`
  
  **haystack-inference@**:
  - `roles/storage.objectAdmin` (for batch-io bucket)

- [ ] **1.5.3** Document service account permissions
- [ ] **1.5.4** Set up Secret Manager access for each account

**Acceptance Criteria**:
- All service accounts created with minimal permissions
- Permissions documented and auditable

---

## Phase 2: Database Schema Implementation

> **Spec Reference**: `./specification/database.md` (Section 8.2: Database Schema)

### Task 2.1: Create Core Database Schema

**Description**: Implement the complete database schema as specified.

- [ ] **2.1.1** Create schema initialization script (`scripts/init_schema.sql`):

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core cells table (~10M rows)
CREATE TABLE cells (
    id SERIAL PRIMARY KEY,
    cell_index INT NOT NULL,
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
    tissue_name VARCHAR(256),
    donor_id VARCHAR(64),
    disease_mondo_id VARCHAR(32),
    disease_name VARCHAR(256),
    sample_condition VARCHAR(256),
    sample_metadata JSONB DEFAULT '{}',
    
    -- External IDs
    perturbation_external_ids JSONB DEFAULT '{}',
    perturbation_targets TEXT[],
    perturbation_pathways TEXT[],
    
    -- Quality metrics
    n_genes INT,
    total_counts FLOAT,
    
    -- Text embeddings (OpenAI text-embedding-3-small, 1536 dim)
    perturbation_embedding vector(1536),
    cell_type_embedding vector(1536),
    sample_context_embedding vector(1536),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **2.1.2** Create donors table:

```sql
CREATE TABLE donors (
    donor_id VARCHAR(64) PRIMARY KEY,
    dataset VARCHAR(32) NOT NULL,
    age_category VARCHAR(32),
    sex VARCHAR(16),
    disease_states TEXT[],
    disease_names TEXT[],
    tissue_types TEXT[],
    n_cells INT,
    cell_types_present TEXT[],
    clinical_embedding vector(1536)
);
```

- [ ] **2.1.3** Create perturbations lookup table:

```sql
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
```

- [ ] **2.1.4** Create cell_types lookup table:

```sql
CREATE TABLE cell_types (
    cell_type_cl_id VARCHAR(32) PRIMARY KEY,
    cell_type_name VARCHAR(256) NOT NULL,
    lineage_cl_ids TEXT[],
    lineage_names TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    perturbations_present TEXT[]
);
```

- [ ] **2.1.5** Create conditions table:

```sql
CREATE TABLE conditions (
    condition_id VARCHAR(64) PRIMARY KEY,
    condition_type VARCHAR(32) NOT NULL,
    perturbation_name VARCHAR(256),
    perturbation_type VARCHAR(32),
    targets TEXT[],
    pathways TEXT[],
    disease_mondo_id VARCHAR(32),
    tissue_uberon_id VARCHAR(32),
    clinical_attributes JSONB DEFAULT '{}',
    condition_embedding vector(1536)
);
```

- [ ] **2.1.6** Create synonyms table:

```sql
CREATE TABLE synonyms (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(256) NOT NULL,
    synonym VARCHAR(256) NOT NULL,
    entity_type VARCHAR(32) NOT NULL  -- 'perturbation' or 'cell_type'
);
```

**Acceptance Criteria**:
- All tables created successfully
- Schema matches specification exactly
- Vector columns support 1536 dimensions

---

### Task 2.2: Create Ontology Tables

**Description**: Create tables for Cell Ontology storage.

> **Spec Reference**: `./specification/ontology-resolution.md` (Section 4: Database Schema)

- [ ] **2.2.1** Create ontology_terms table:

```sql
CREATE TABLE ontology_terms (
    id SERIAL PRIMARY KEY,
    term_id VARCHAR(32) NOT NULL,
    name VARCHAR(512) NOT NULL,
    definition TEXT,
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_ontology_terms_id_version UNIQUE (term_id, ontology_type, version)
);

COMMENT ON TABLE ontology_terms IS 'Cell Ontology terms with embeddings';
COMMENT ON COLUMN ontology_terms.embedding IS 'OpenAI text-embedding-3-small vector (1536 dim)';
```

- [ ] **2.2.2** Create ontology_relationships table:

```sql
CREATE TABLE ontology_relationships (
    id SERIAL PRIMARY KEY,
    subject_term_id VARCHAR(32) NOT NULL,
    object_term_id VARCHAR(32) NOT NULL,
    relationship_type VARCHAR(64) NOT NULL,  -- is_a, part_of, develops_from
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_ontology_rels UNIQUE (subject_term_id, object_term_id, relationship_type, ontology_type, version)
);

COMMENT ON TABLE ontology_relationships IS 'Cell Ontology relationship graph';
```

**Acceptance Criteria**:
- Ontology tables created with correct constraints
- Supports versioned ontology data

---

### Task 2.3: Create Run History Table

**Description**: Create the runs table for tracking HAYSTACK executions.

> **Spec Reference**: `./specification/data-models.md` (HaystackRun model)

- [ ] **2.3.1** Create runs table:

```sql
CREATE TABLE runs (
    run_id VARCHAR(64) PRIMARY KEY,
    user_email VARCHAR(256),
    status VARCHAR(32) NOT NULL,  -- pending, running, completed, failed, cancelled
    batch_job_name VARCHAR(256),
    control_strategy VARCHAR(32) DEFAULT 'synthetic_control',
    control_strategy_effective VARCHAR(32),
    control_cells_available BOOLEAN DEFAULT FALSE,
    
    raw_query TEXT NOT NULL,
    structured_query JSONB,
    config JSONB NOT NULL,
    random_seed INT,
    
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    current_phase VARCHAR(32),
    current_iteration INT DEFAULT 0,
    
    iterations JSONB DEFAULT '[]',
    final_score INT,
    termination_reason VARCHAR(256),
    error_message TEXT,
    
    output_anndata_path VARCHAR(512),
    output_report_path VARCHAR(512),
    output_log_path VARCHAR(512),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Acceptance Criteria**:
- Runs table created with all required columns
- Supports control strategy tracking

---

### Task 2.4: Create Database Indexes

**Description**: Create all required indexes for query performance.

> **Spec Reference**: `./specification/database.md` (Section 8.2: Indexes)

- [ ] **2.4.1** Create cell table indexes:

```sql
-- Standard indexes
CREATE INDEX idx_cells_dataset ON cells(dataset);
CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
CREATE INDEX idx_cells_tissue ON cells(tissue_uberon_id);
CREATE INDEX idx_cells_is_control ON cells(is_control);
CREATE INDEX idx_cells_donor ON cells(donor_id);
CREATE INDEX idx_cells_disease ON cells(disease_mondo_id);
CREATE INDEX idx_cells_condition ON cells(sample_condition);

-- Composite indexes for common query patterns
CREATE INDEX idx_cells_pert_ct_donor ON cells(perturbation_name, cell_type_cl_id, donor_id);
CREATE INDEX idx_cells_tissue_ct ON cells(tissue_uberon_id, cell_type_cl_id);
CREATE INDEX idx_cells_donor_ct ON cells(donor_id, cell_type_cl_id);
```

- [ ] **2.4.2** Create HNSW vector indexes (defer until after data load):

```sql
-- NOTE: Build AFTER loading data for best performance
CREATE INDEX idx_cells_perturbation_embedding ON cells 
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_cell_type_embedding ON cells 
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_sample_context_embedding ON cells 
USING hnsw (sample_context_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- [ ] **2.4.3** Create ontology table indexes:

```sql
CREATE INDEX idx_ontology_terms_term_id ON ontology_terms(term_id);
CREATE INDEX idx_ontology_terms_name ON ontology_terms(name);
CREATE INDEX idx_ontology_terms_name_lower ON ontology_terms(LOWER(name));
CREATE INDEX idx_ontology_terms_type_version ON ontology_terms(ontology_type, version);

CREATE INDEX idx_ontology_terms_embedding ON ontology_terms
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_ontology_rels_subject ON ontology_relationships(subject_term_id);
CREATE INDEX idx_ontology_rels_object ON ontology_relationships(object_term_id);
CREATE INDEX idx_ontology_rels_type ON ontology_relationships(relationship_type);
CREATE INDEX idx_ontology_rels_subject_version ON ontology_relationships(
    subject_term_id, ontology_type, version);
CREATE INDEX idx_ontology_rels_object_version ON ontology_relationships(
    object_term_id, ontology_type, version);
```

- [ ] **2.4.4** Create other indexes:

```sql
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
CREATE INDEX idx_synonyms_synonym_lower ON synonyms(LOWER(synonym));
CREATE INDEX idx_synonyms_type ON synonyms(entity_type);
CREATE INDEX idx_perturbations_targets ON perturbations USING GIN(targets);
CREATE INDEX idx_perturbations_pathways ON perturbations USING GIN(pathways);
CREATE INDEX idx_cell_types_lineage ON cell_types USING GIN(lineage_cl_ids);
CREATE INDEX idx_donors_dataset ON donors(dataset);
CREATE INDEX idx_conditions_type ON conditions(condition_type);
CREATE INDEX idx_runs_user ON runs(user_email);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at DESC);
```

**Acceptance Criteria**:
- All indexes created successfully
- Query performance validated with EXPLAIN ANALYZE

---

### Task 2.5: Create Database Roles

**Description**: Set up database roles with appropriate permissions.

> **Spec Reference**: `./specification/database.md` (Section 8.3: Database Roles)

- [ ] **2.5.1** Create application role:

```sql
CREATE ROLE haystack_app WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_app;
GRANT USAGE ON SCHEMA public TO haystack_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON runs, synonyms TO haystack_app;
GRANT SELECT ON cells, donors, perturbations, cell_types, conditions, 
    ontology_terms, ontology_relationships TO haystack_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO haystack_app;
```

- [ ] **2.5.2** Create agent role with read-only access:

```sql
CREATE ROLE haystack_agent WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_agent;
GRANT USAGE ON SCHEMA public TO haystack_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO haystack_agent;

-- Set statement timeout for safety
ALTER ROLE haystack_agent SET statement_timeout = '30s';
```

- [ ] **2.5.3** Create admin role:

```sql
CREATE ROLE haystack_admin WITH LOGIN PASSWORD 'secure_password' SUPERUSER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO haystack_admin;
```

- [ ] **2.5.4** Store all role passwords in Secret Manager

**Acceptance Criteria**:
- All roles created with minimal permissions
- Passwords stored securely in Secret Manager
- Agent role has 30s statement timeout

---

## Phase 3: Atlas Data Processing Pipeline

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 2: Atlas Data Model)

### Task 3.1: Download Atlas Datasets

**Description**: Obtain the three atlas datasets used by HAYSTACK.

- [ ] **3.1.1** Download Parse PBMC dataset:
  - Source: Parse Biosciences / GEO
  - Size: ~10M cells, 90 cytokines, 12 donors
  - Format: H5AD

- [ ] **3.1.2** Download OpenProblems dataset:
  - Source: OpenProblems benchmark
  - Size: ~500K cells, 147 drugs, 3 donors
  - Format: H5AD

- [ ] **3.1.3** Download Tabula Sapiens dataset:
  - Source: Tabula Sapiens consortium
  - Size: ~500K cells, unperturbed, 25 tissues, 24 donors
  - Format: H5AD

- [ ] **3.1.4** Upload raw H5AD files to GCS (`haystack-atlases/raw/`)

**Acceptance Criteria**:
- All datasets downloaded and validated
- Files uploaded to GCS with proper checksums

---

### Task 3.2: Implement Harmonization Pipeline

**Description**: Create the data harmonization pipeline to standardize metadata.

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 2.3: Dataset-Specific Harmonization)

- [ ] **3.2.1** Create harmonization script (`scripts/harmonize_atlas.py`):

```python
"""
Atlas harmonization pipeline.

Harmonizes cell metadata across Parse PBMC, OpenProblems, and Tabula Sapiens
to a common schema for indexing.
"""

import scanpy as sc
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class HarmonizedCellMetadata(BaseModel):
    """Harmonized metadata for a single cell."""
    cell_index: int
    dataset: str
    
    # Cell type
    cell_type_original: str
    cell_type_cl_id: Optional[str] = None
    cell_type_name: Optional[str] = None
    
    # Perturbation
    perturbation_original: Optional[str] = None
    perturbation_name: Optional[str] = None
    perturbation_type: Optional[str] = None
    is_control: bool
    
    # Tissue
    tissue_original: Optional[str] = None
    tissue_uberon_id: Optional[str] = None
    tissue_name: Optional[str] = None
    
    # Donor
    donor_id: str
    
    # Disease
    disease_mondo_id: Optional[str] = None
    disease_name: Optional[str] = None
    
    # Sample
    sample_condition: Optional[str] = None
    sample_metadata: dict = {}
    
    # Quality
    n_genes: Optional[int] = None
    total_counts: Optional[float] = None
```

- [ ] **3.2.2** Implement Parse PBMC harmonization:

| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_type` | `cell_type_original` | Direct copy |
| `cell_type` | `cell_type_cl_id` | Lookup in CL mapping |
| `stim` | `perturbation_original` | Direct copy |
| `stim` | `perturbation_name` | Normalize (e.g., "IFNg" → "IFN-gamma") |
| `donor` | `donor_id` | Prefix "parse_" |

- [ ] **3.2.3** Implement OpenProblems harmonization:

| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_type` | `cell_type_original` | Direct copy |
| `cell_type` | `cell_type_cl_id` | Direct (already has CL) |
| `sm_name` | `perturbation_original` | Direct copy |
| `sm_name` | `perturbation_name` | PubChem/ChEMBL resolution |
| N/A | `perturbation_type` | Set to "drug" |
| `donor_id` | `donor_id` | Prefix "op_" |

- [ ] **3.2.4** Implement Tabula Sapiens harmonization:

| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_type_ontology_term_id` | `cell_type_cl_id` | Direct copy |
| `cell_type` | `cell_type_name` | Direct copy |
| `tissue` | `tissue_original` | Direct copy |
| `tissue_ontology_term_id` | `tissue_uberon_id` | Direct copy |
| N/A | `perturbation_name` | Set to None |
| N/A | `is_control` | Set to True |
| `donor` | `donor_id` | Prefix "ts_" |

- [ ] **3.2.5** Create cell type mapping file (`data/cell_type_mappings.csv`):
  - Map common cell type strings to CL IDs
  - Include synonyms

- [ ] **3.2.6** Create perturbation normalization file (`data/perturbation_mappings.csv`):
  - Map cytokine abbreviations to full names
  - External IDs (PubChem, DrugBank, ChEMBL)

**Acceptance Criteria**:
- All three datasets harmonized to common schema
- Cell types mapped to CL IDs where possible
- Perturbations normalized with external IDs

---

### Task 3.3: Generate Text Embeddings

**Description**: Generate OpenAI embeddings for semantic search.

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 3.3: Text Embedding Generation)

- [ ] **3.3.1** Create embedding generation script (`scripts/generate_embeddings.py`):

```python
"""
Generate text embeddings for cell metadata.

Uses OpenAI text-embedding-3-small (1536 dimensions) for:
- perturbation_embedding: Perturbation name + type + targets
- cell_type_embedding: Cell type name + lineage
- sample_context_embedding: Tissue + disease + condition
"""

import openai
from typing import list

async def generate_perturbation_embedding(
    perturbation_name: str,
    perturbation_type: str,
    targets: list[str],
    pathways: list[str],
) -> list[float]:
    """Generate embedding for perturbation context."""
    text = f"Perturbation: {perturbation_name}"
    if perturbation_type:
        text += f" ({perturbation_type})"
    if targets:
        text += f" targeting {', '.join(targets[:5])}"
    if pathways:
        text += f" affecting {', '.join(pathways[:3])} pathways"
    
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
```

- [ ] **3.3.2** Generate perturbation embeddings for all unique perturbations
- [ ] **3.3.3** Generate cell type embeddings for all unique cell types
- [ ] **3.3.4** Generate sample context embeddings for all unique contexts
- [ ] **3.3.5** Implement batching (max 2048 texts per API call)
- [ ] **3.3.6** Implement rate limiting (3000 RPM)

**Acceptance Criteria**:
- All embeddings generated and stored
- Embeddings are 1536 dimensions
- Rate limits respected

---

### Task 3.4: Build Database Population Script

**Description**: Load harmonized data into Cloud SQL.

- [ ] **3.4.1** Create population script (`scripts/populate_database.py`):

```python
"""
Populate Cloud SQL database with harmonized atlas data.
"""

import asyncpg
import asyncio
from tqdm import tqdm

BATCH_SIZE = 10000

async def insert_cells(pool, cells: list[dict]):
    """Batch insert cells to database."""
    async with pool.acquire() as conn:
        await conn.executemany('''
            INSERT INTO cells (
                cell_index, dataset, cell_type_original, cell_type_cl_id,
                cell_type_name, perturbation_original, perturbation_name,
                perturbation_type, is_control, tissue_original, tissue_uberon_id,
                tissue_name, donor_id, disease_mondo_id, disease_name,
                sample_condition, sample_metadata, perturbation_external_ids,
                perturbation_targets, perturbation_pathways, n_genes, total_counts,
                perturbation_embedding, cell_type_embedding, sample_context_embedding
            ) VALUES ($1, $2, $3, ...)
        ''', [cell.values() for cell in cells])
```

- [ ] **3.4.2** Implement connection pooling for performance
- [ ] **3.4.3** Add progress reporting with tqdm
- [ ] **3.4.4** Handle partial failures gracefully
- [ ] **3.4.5** Populate lookup tables (donors, perturbations, cell_types)

**Expected Duration**: 2-4 hours for ~10M cells

**Acceptance Criteria**:
- All cells inserted successfully
- Lookup tables populated with aggregations
- Script handles interruption gracefully

---

### Task 3.5: Build HNSW Indexes

**Description**: Build vector similarity indexes after data load.

- [ ] **3.5.1** Create index building script (`scripts/build_indexes.sql`):

```sql
-- Build HNSW indexes after data is loaded
-- This is more efficient than building during inserts

SET maintenance_work_mem = '4GB';

CREATE INDEX CONCURRENTLY idx_cells_perturbation_embedding 
ON cells USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_cells_cell_type_embedding 
ON cells USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_cells_sample_context_embedding 
ON cells USING hnsw (sample_context_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Update statistics
ANALYZE cells;
```

- [ ] **3.5.2** Run VACUUM ANALYZE after index creation
- [ ] **3.5.3** Verify index usage with EXPLAIN ANALYZE

**Expected Duration**: 30-60 minutes per index

**Acceptance Criteria**:
- All HNSW indexes built successfully
- Vector queries use index (verify with EXPLAIN)

---

## Phase 4: Cell Ontology Integration

> **Spec Reference**: `./specification/ontology-resolution.md`

### Task 4.1: Download and Parse Cell Ontology

**Description**: Obtain and parse the Cell Ontology OBO file.

- [ ] **4.1.1** Download CL OBO file from OBO Foundry:
  ```bash
  wget http://purl.obolibrary.org/obo/cl.obo -O data/cl.obo
  ```

- [ ] **4.1.2** Parse OBO file to extract terms and relationships:

```python
"""
Parse Cell Ontology OBO file.
"""

import pronto

def parse_cell_ontology(obo_path: str) -> tuple[list, list]:
    """Parse CL OBO file to terms and relationships."""
    onto = pronto.Ontology(obo_path)
    
    terms = []
    relationships = []
    
    for term in onto.terms():
        if term.id.startswith("CL:"):
            terms.append({
                "term_id": term.id,
                "name": term.name,
                "definition": term.definition.strip() if term.definition else None,
            })
            
            # Extract is_a relationships
            for parent in term.superclasses(distance=1):
                if parent.id.startswith("CL:") and parent.id != term.id:
                    relationships.append({
                        "subject_term_id": term.id,
                        "object_term_id": parent.id,
                        "relationship_type": "is_a",
                    })
    
    return terms, relationships
```

- [ ] **4.1.3** Extract all term attributes:
  - term_id, name, definition
  - synonyms (for synonym table)
  
- [ ] **4.1.4** Extract relationships:
  - is_a, part_of, develops_from, has_part

**Acceptance Criteria**:
- All CL terms parsed (~2500 terms)
- All relationships extracted
- Synonyms captured

---

### Task 4.2: Generate Ontology Embeddings

**Description**: Generate embeddings for all CL terms.

- [ ] **4.2.1** Generate embeddings for term names + definitions:

```python
async def generate_ontology_embeddings(terms: list[dict]) -> list[list[float]]:
    """Generate embeddings for ontology terms."""
    texts = []
    for term in terms:
        text = term["name"]
        if term.get("definition"):
            text += f": {term['definition']}"
        texts.append(text)
    
    # Batch embedding generation
    embeddings = await batch_embed(texts, model="text-embedding-3-small")
    return embeddings
```

- [ ] **4.2.2** Batch process to respect rate limits
- [ ] **4.2.3** Store embeddings with version tag

**Acceptance Criteria**:
- All terms have embeddings
- Version tracked for ontology updates

---

### Task 4.3: Populate Ontology Tables

**Description**: Load ontology data into Cloud SQL.

- [ ] **4.3.1** Create ontology loading script (`scripts/load_ontology.py`):

```python
async def load_ontology(version: str):
    """Load Cell Ontology into database."""
    terms, relationships = parse_cell_ontology("data/cl.obo")
    embeddings = await generate_ontology_embeddings(terms)
    
    async with database.connection() as conn:
        # Insert terms
        for term, embedding in zip(terms, embeddings):
            await conn.execute('''
                INSERT INTO ontology_terms 
                (term_id, name, definition, ontology_type, version, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (term_id, ontology_type, version) 
                DO UPDATE SET name = $2, definition = $3, embedding = $6
            ''', term["term_id"], term["name"], term["definition"], 
                "cell", version, embedding)
        
        # Insert relationships
        for rel in relationships:
            await conn.execute('''
                INSERT INTO ontology_relationships
                (subject_term_id, object_term_id, relationship_type,
                 ontology_type, version)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
            ''', rel["subject_term_id"], rel["object_term_id"],
                rel["relationship_type"], "cell", version)
```

- [ ] **4.3.2** Build HNSW index for ontology embeddings
- [ ] **4.3.3** Verify graph connectivity

**Acceptance Criteria**:
- All terms and relationships loaded
- HNSW index built for semantic search
- Graph traversal works correctly

---

### Task 4.4: Populate Cell Type Lineage

**Description**: Pre-compute and store cell type lineage information.

> **Spec Reference**: `./specification/database.md` (Section 8.2.1: Cell Type Lineage Population)

- [ ] **4.4.1** Create lineage population script:

```python
async def populate_cell_type_lineage():
    """Pre-compute and store lineage for all cell types."""
    async with database.connection() as conn:
        # Get all unique cell types
        cell_types = await conn.fetch(
            "SELECT DISTINCT cell_type_cl_id FROM cells WHERE cell_type_cl_id IS NOT NULL"
        )
        
        for ct in cell_types:
            cl_id = ct["cell_type_cl_id"]
            
            # Get ancestors via ontology
            ancestors = await get_ancestors(cl_id)
            lineage_ids = [a["term_id"] for a in ancestors]
            lineage_names = [a["name"] for a in ancestors]
            
            # Update cell_types table
            await conn.execute('''
                UPDATE cell_types
                SET lineage_cl_ids = $1, lineage_names = $2
                WHERE cell_type_cl_id = $3
            ''', lineage_ids, lineage_names, cl_id)
```

- [ ] **4.4.2** Run for all unique cell types in database
- [ ] **4.4.3** Verify lineage correctness with spot checks

**Acceptance Criteria**:
- All cell types have lineage populated
- Lineage chain is correct and complete

---

## Phase 5: Validation and Documentation

### Task 5.1: Validate Data Quality

**Description**: Ensure data integrity and completeness.

- [ ] **5.1.1** Verify cell counts match expected:
  - Parse PBMC: ~10M cells
  - OpenProblems: ~500K cells
  - Tabula Sapiens: ~500K cells

- [ ] **5.1.2** Verify CL ID coverage:
  - Target: >90% of cells have CL ID

- [ ] **5.1.3** Verify embedding completeness:
  - All non-null perturbations have embeddings
  - All cell types have embeddings

- [ ] **5.1.4** Run sample queries to verify indexes:

```sql
-- Test vector search
EXPLAIN ANALYZE
SELECT cell_index, perturbation_name,
       perturbation_embedding <=> '[query_vector]' as distance
FROM cells
ORDER BY perturbation_embedding <=> '[query_vector]'
LIMIT 10;

-- Verify index usage
-- Should see "Index Scan using idx_cells_perturbation_embedding"
```

**Acceptance Criteria**:
- Cell counts match expected
- CL ID coverage >90%
- All indexes operational

---

### Task 5.2: Create Development Seed Data

**Description**: Create a smaller dataset for local development.

- [ ] **5.2.1** Create seed data script (`scripts/seed_db.py`):
  - Sample 10K cells from each dataset (30K total)
  - Include diverse perturbations and cell types
  - Include all ontology data

- [ ] **5.2.2** Create Docker Compose for local PostgreSQL:

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: haystack
      POSTGRES_USER: haystack
      POSTGRES_PASSWORD: devpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
      - ./scripts/seed_data.sql:/docker-entrypoint-initdb.d/02-seed.sql

volumes:
  postgres_data:
```

**Acceptance Criteria**:
- Local development environment works with seed data
- All queries testable locally

---

### Task 5.3: Document Infrastructure

**Description**: Create comprehensive infrastructure documentation.

- [ ] **5.3.1** Document GCP resource inventory
- [ ] **5.3.2** Document database schema with ERD
- [ ] **5.3.3** Document data pipeline with flow diagram
- [ ] **5.3.4** Document permissions and access patterns
- [ ] **5.3.5** Create runbook for common operations:
  - Adding new atlas data
  - Updating Cell Ontology
  - Database maintenance

**Acceptance Criteria**:
- All infrastructure documented
- Runbooks tested and verified

---

## Definition of Done

- [ ] All GCP resources provisioned and documented
- [ ] Database schema created with all tables and indexes
- [ ] All three atlas datasets harmonized and loaded (~10M cells)
- [ ] Cell Ontology loaded with embeddings (~2500 terms)
- [ ] Vector indexes built and verified
- [ ] Local development environment functional
- [ ] Infrastructure documentation complete
- [ ] Data quality validation passed

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Atlas download failures | High | Use multiple download sources, cache locally |
| Embedding API rate limits | Medium | Implement exponential backoff, batch requests |
| Database load performance | Medium | Use COPY instead of INSERT, disable indexes during load |
| VPC configuration issues | High | Follow GCP documentation exactly, test early |
| Cell type mapping gaps | Medium | Use semantic search as fallback, log unmapped types |

---

## Estimated Costs

| Resource | Monthly Cost (Estimate) |
|----------|------------------------|
| Cloud SQL (db-custom-4-15360) | ~$300/month |
| GCS Storage (200 GB) | ~$5/month |
| OpenAI Embeddings (one-time) | ~$50 (10M cells) |
| VPC Connector | ~$30/month |

**Total Setup Cost**: ~$50 (one-time) + ~$335/month ongoing

---

## Next Sprint

**Sprint 02: Core Backend Services** - Implement the database client, GCS service, and external API integrations.
