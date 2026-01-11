# Database Specification

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
    tissue_name VARCHAR(256),
    donor_id VARCHAR(64),
    disease_mondo_id VARCHAR(32),
    disease_name VARCHAR(256),
    sample_condition VARCHAR(256),
    sample_metadata JSONB DEFAULT '{}',
    
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
    sample_context_embedding vector(1536),
    
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
    tissue_uberon_id VARCHAR(32),
    tissue_name VARCHAR(256),
    disease_mondo_id VARCHAR(32),
    disease_name VARCHAR(256),
    sample_condition VARCHAR(256),
    sample_metadata JSONB DEFAULT '{}',
    is_reference_sample BOOLEAN DEFAULT FALSE,
    
    n_cells INT NOT NULL,
    cell_indices INT[] NOT NULL,
    
    mean_n_genes FLOAT,
    mean_total_counts FLOAT,
    
    has_control BOOLEAN DEFAULT FALSE,
    control_group_id VARCHAR(64),
    
    -- Representative embeddings (mean of cells in group)
    perturbation_embedding vector(1536),
    cell_type_embedding vector(1536),
    sample_context_embedding vector(1536),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Donor lookup table
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

-- Unified conditions table
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

-- Cell type lookup table
CREATE TABLE cell_types (
    cell_type_cl_id VARCHAR(32) PRIMARY KEY,
    cell_type_name VARCHAR(256) NOT NULL,
    lineage_cl_ids TEXT[],
    lineage_names TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    perturbations_present TEXT[]
);


-- =============================================================================
-- CELL ONTOLOGY TABLES
-- =============================================================================
-- These tables store Cell Ontology terms and relationships for semantic search
-- and graph traversal. Data is pre-computed and loaded from GCS.

-- Ontology terms table with vector embeddings
CREATE TABLE ontology_terms (
    id SERIAL PRIMARY KEY,
    
    -- Term identifiers
    term_id VARCHAR(32) NOT NULL,
    name VARCHAR(512) NOT NULL,
    definition TEXT,
    
    -- Ontology metadata
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    
    -- Vector embedding for semantic search
    -- OpenAI text-embedding-3-small produces 1536-dimensional vectors
    embedding vector(1536),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT uq_ontology_terms_id_version 
        UNIQUE (term_id, ontology_type, version)
);

COMMENT ON TABLE ontology_terms IS 'Cell Ontology terms with embeddings for semantic search';
COMMENT ON COLUMN ontology_terms.term_id IS 'Cell Ontology ID (e.g., CL:0000057)';
COMMENT ON COLUMN ontology_terms.embedding IS 'OpenAI text-embedding-3-small vector (1536 dim)';
COMMENT ON COLUMN ontology_terms.version IS 'Ontology version in YYYY-MM-DD format';


-- Ontology relationships table
CREATE TABLE ontology_relationships (
    id SERIAL PRIMARY KEY,
    
    -- Relationship endpoints
    subject_term_id VARCHAR(32) NOT NULL,
    object_term_id VARCHAR(32) NOT NULL,
    relationship_type VARCHAR(64) NOT NULL,
    
    -- Ontology metadata
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT uq_ontology_rels 
        UNIQUE (subject_term_id, object_term_id, relationship_type, ontology_type, version)
);

COMMENT ON TABLE ontology_relationships IS 'Cell Ontology relationships (is_a, part_of, develops_from, etc.)';
COMMENT ON COLUMN ontology_relationships.subject_term_id IS 'Source term ID (child in is_a)';
COMMENT ON COLUMN ontology_relationships.object_term_id IS 'Target term ID (parent in is_a)';
COMMENT ON COLUMN ontology_relationships.relationship_type IS 'Relationship type: is_a, part_of, develops_from, etc.)';

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
CREATE INDEX idx_cells_disease ON cells(disease_mondo_id);
CREATE INDEX idx_cells_condition ON cells(sample_condition);

CREATE INDEX idx_cell_types_lineage ON cell_types USING GIN(lineage_cl_ids);

-- HNSW vector indexes for similarity search
CREATE INDEX idx_cells_perturbation_embedding ON cells 
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_cell_type_embedding ON cells 
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_sample_context_embedding ON cells 
USING hnsw (sample_context_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- ONTOLOGY INDEXES
-- =============================================================================

-- Term lookup indexes
CREATE INDEX idx_ontology_terms_term_id ON ontology_terms(term_id);
CREATE INDEX idx_ontology_terms_name ON ontology_terms(name);
CREATE INDEX idx_ontology_terms_name_lower ON ontology_terms(LOWER(name));
CREATE INDEX idx_ontology_terms_type_version ON ontology_terms(ontology_type, version);

-- HNSW vector index for semantic search
-- IMPORTANT: Build AFTER loading data for best performance
CREATE INDEX idx_ontology_terms_embedding ON ontology_terms
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Relationship indexes for graph traversal
CREATE INDEX idx_ontology_rels_subject ON ontology_relationships(subject_term_id);
CREATE INDEX idx_ontology_rels_object ON ontology_relationships(object_term_id);
CREATE INDEX idx_ontology_rels_type ON ontology_relationships(relationship_type);
CREATE INDEX idx_ontology_rels_version ON ontology_relationships(ontology_type, version);

-- Composite index for efficient neighbor queries
CREATE INDEX idx_ontology_rels_subject_version ON ontology_relationships(
    subject_term_id, ontology_type, version
);
CREATE INDEX idx_ontology_rels_object_version ON ontology_relationships(
    object_term_id, ontology_type, version
);

-- Index on synonym table
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
CREATE INDEX idx_synonyms_type ON synonyms(entity_type);

CREATE INDEX idx_donors_dataset ON donors(dataset);
CREATE INDEX idx_conditions_type ON conditions(condition_type);

-- Index on runs table
CREATE INDEX idx_runs_user ON runs(user_email);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created ON runs(created_at DESC);
```

### 8.2.1 Cell Type Lineage Population

```python
# scripts/populate_cell_type_lineage.py
"""Populate cell type lineage from Cell Ontology."""

async def populate_lineage():
    """Pre-compute and store lineage for all cell types."""
    from haystack.orchestrator.services.ontology import CellOntologyService
    from haystack.orchestrator.services.database import HaystackDatabase
    
    ontology = CellOntologyService.get_instance()
    db = HaystackDatabase.get_instance()
    
    # Get all distinct cell types
    cell_types = await db.execute_query(
        "SELECT DISTINCT cell_type_cl_id FROM cell_types WHERE cell_type_cl_id IS NOT NULL"
    )
    
    for row in cell_types:
        cl_id = row["cell_type_cl_id"]
        
        # Get lineage from ontology
        lineage_names = await ontology.get_lineage(cl_id, max_depth=5)
        
        # Get lineage CL IDs (need to resolve names to IDs)
        lineage_ids = []
        for name in lineage_names:
            results = await ontology.semantic_search([name], k=1, distance_threshold=0.3)
            if results.get(name) and isinstance(results[name], list):
                lineage_ids.append(results[name][0]["term_id"])
        
        # Update cell_types table
        await db.execute_query(
            """
            UPDATE cell_types 
            SET lineage_cl_ids = $1, lineage_names = $2
            WHERE cell_type_cl_id = $3
            """,
            (lineage_ids, lineage_names, cl_id)
        )
    
    print(f"Updated lineage for {len(cell_types)} cell types")
```

### 8.3 Database Roles

```sql
-- Application role (read-write for runs, read-only for cells)
CREATE ROLE haystack_app WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_app;
GRANT USAGE ON SCHEMA public TO haystack_app;
GRANT SELECT ON cells, cell_groups, donors, conditions, perturbations, cell_types, synonyms TO haystack_app;
GRANT SELECT ON ontology_terms, ontology_relationships TO haystack_app;
GRANT SELECT, INSERT, UPDATE ON runs TO haystack_app;
GRANT USAGE, SELECT ON SEQUENCE runs_run_id_seq TO haystack_app;

-- Agent role (read-only for all tables)
CREATE ROLE haystack_agent WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haystack TO haystack_agent;
GRANT USAGE ON SCHEMA public TO haystack_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO haystack_agent;
GRANT SELECT ON ontology_terms, ontology_relationships TO haystack_agent;
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

## Related Specs

- `specification/data-models.md`
- `specification/backend-api.md`
