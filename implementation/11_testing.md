# Sprint 11: Testing

**Duration**: 2 weeks  
**Dependencies**: All previous sprints  
**Goal**: Implement comprehensive test suite for all components.

---

## Overview

> **Spec Reference**: `./specification/testing.md`

This sprint implements:
- Backend unit tests
- Frontend component tests
- Integration tests
- Load testing
- Cell Ontology integration tests

---

## Phase 1: Test Infrastructure

### Task 1.1: Set Up pytest Configuration

- [ ] **1.1.1** Create `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=. --cov-report=html --cov-report=term-missing
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow running
```

- [ ] **1.1.2** Create `tests/conftest.py`:

```python
import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    from api.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def mock_database():
    mock = AsyncMock()
    mock.get_run.return_value = {
        "run_id": "test_123",
        "status": "running",
        "iterations": [],
    }
    return mock

@pytest.fixture
def mock_gcs():
    return AsyncMock()

@pytest.fixture
def mock_batch_client():
    return AsyncMock()
```

---

## Phase 2: Backend Unit Tests

### Task 2.1: API Endpoint Tests

> **Spec Reference**: `./specification/testing.md` (Section 15.1)

- [ ] **2.1.1** Create `tests/api/test_runs.py`:

```python
import pytest
from httpx import AsyncClient

class TestRunEndpoints:
    
    @pytest.mark.asyncio
    async def test_create_run(self, client: AsyncClient, mock_database):
        response = await client.post(
            "/api/v1/runs/",
            json={"query": "How would lung fibroblasts respond to TGF-beta?"},
            headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:test@example.com"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_create_run_invalid_query(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/runs/",
            json={"query": "short"},  # Below 10 char minimum
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_get_run_status(self, client: AsyncClient, mock_database):
        response = await client.get("/api/v1/runs/test_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test_123"
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_get_run_not_found(self, client: AsyncClient, mock_database):
        mock_database.get_run.return_value = None
        response = await client.get("/api/v1/runs/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_cancel_run(self, client: AsyncClient, mock_database):
        response = await client.post("/api/v1/runs/test_123/cancel")
        assert response.status_code == 200
```

---

### Task 2.2: Retrieval Strategy Tests

- [ ] **2.2.1** Create `tests/orchestrator/test_retrieval.py`:

```python
import pytest
from orchestrator.retrieval.direct_match import DirectMatchStrategy
from orchestrator.retrieval.semantic_match import SemanticMatchStrategy
from orchestrator.retrieval.ranker import CandidateRanker

class TestDirectMatchStrategy:
    
    @pytest.mark.asyncio
    async def test_exact_match(self, mock_database):
        mock_database.execute_query.return_value = [
            {
                "dataset": "parse_pbmc",
                "perturbation_name": "TGF-beta",
                "cell_type_cl_id": "CL:0002553",
                "cell_indices": [1, 2, 3],
                "n_cells": 3,
            }
        ]
        
        strategy = DirectMatchStrategy(mock_database)
        results = await strategy.retrieve(
            query=mock_structured_query(perturbation="TGF-beta", cell_type="CL:0002553"),
            max_results=10,
        )
        
        assert len(results) == 1
        assert results[0].perturbation_name == "TGF-beta"
    
    @pytest.mark.asyncio
    async def test_no_match(self, mock_database):
        mock_database.execute_query.return_value = []
        
        strategy = DirectMatchStrategy(mock_database)
        results = await strategy.retrieve(
            query=mock_structured_query(perturbation="unknown", cell_type="CL:0000000"),
            max_results=10,
        )
        
        assert len(results) == 0

class TestCandidateRanker:
    
    def test_ranking_weights(self):
        ranker = CandidateRanker(
            relevance_weight=0.4,
            quality_weight=0.3,
            diversity_weight=0.3,
        )
        
        # Verify weights sum to 1.0
        total = ranker.relevance_weight + ranker.quality_weight + ranker.diversity_weight
        assert total == 1.0
    
    def test_diversity_penalty(self):
        ranker = CandidateRanker()
        
        # First candidate should have max diversity
        score1 = ranker._compute_diversity_score(mock_candidate("A"), [])
        assert score1 == 1.0
        
        # Second same-type candidate should be penalized
        selected = [mock_candidate("A")]
        score2 = ranker._compute_diversity_score(mock_candidate("A"), selected)
        assert score2 < 1.0
```

---

### Task 2.3: Ontology Resolution Tests

> **Spec Reference**: `./specification/testing.md` (Section 15.4)

- [ ] **2.3.1** Create `tests/orchestrator/test_ontology.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from orchestrator.agents.query_understanding import resolve_cell_type_with_fallback

class TestCellTypeResolution:
    
    @pytest.mark.asyncio
    async def test_semantic_search_success(self):
        with patch("orchestrator.services.ontology.CellOntologyService") as mock:
            mock.get_instance.return_value.semantic_search.return_value = {
                "fibroblast": [{"term_id": "CL:0000057", "name": "fibroblast", "distance": 0.1}]
            }
            
            result = await resolve_cell_type_with_fallback("fibroblast")
            
            assert result["resolved"] is True
            assert result["term_id"] == "CL:0000057"
            assert result["method"] == "semantic"
    
    @pytest.mark.asyncio
    async def test_ols_fallback(self):
        with patch("orchestrator.services.ontology.CellOntologyService") as mock:
            # Semantic search fails
            mock.get_instance.return_value.semantic_search.return_value = {"rare cell": []}
            # OLS succeeds
            mock.get_instance.return_value.query_ols.return_value = {
                "rare cell": [{"term_id": "CL:0000999", "name": "rare cell type"}]
            }
            
            result = await resolve_cell_type_with_fallback("rare cell")
            
            assert result["resolved"] is True
            assert result["method"] == "ols"
    
    @pytest.mark.asyncio
    async def test_unresolved_cell_type(self):
        with patch("orchestrator.services.ontology.CellOntologyService") as mock:
            mock.get_instance.return_value.semantic_search.return_value = {"unknown": []}
            mock.get_instance.return_value.query_ols.return_value = {"unknown": []}
            
            result = await resolve_cell_type_with_fallback("unknown")
            
            assert result["resolved"] is False
            assert "warning" in result
```

---

### Task 2.4: Control Strategy Tests

> **Spec Reference**: `./specification/testing.md` (Section 15.4)

- [ ] **2.4.1** Create `tests/orchestrator/test_control_strategy.py`:

```python
import pytest
from shared.models.queries import ControlStrategy
from orchestrator.agents.orchestrator import OrchestratorAgent

class TestControlStrategy:
    
    @pytest.mark.asyncio
    async def test_synthetic_control_with_matched_cells(self, mock_database, mock_gcs):
        mock_database.get_run.return_value = {"raw_query": "...", "config": {}}
        
        agent = OrchestratorAgent(
            run_id="test_123",
            query="How would T cells respond to IL-6?",
            user_email="test@example.com",
            config={},
            control_strategy="synthetic_control",
        )
        
        # Mock matched control cells found
        agent.prompt_agent.find_matched_controls = AsyncMock(return_value={
            "donor_id": "donor_1",
            "n_cells": 100,
        })
        
        result = await agent.run()
        
        assert mock_gcs.submit_inference_job.call_count == 2  # Perturbed + control
    
    @pytest.mark.asyncio
    async def test_synthetic_control_fallback(self, mock_database, mock_gcs):
        agent = OrchestratorAgent(
            run_id="test_123",
            query="How would T cells respond to IL-6?",
            user_email="test@example.com",
            config={},
            control_strategy="synthetic_control",
        )
        
        # No matched controls
        agent.prompt_agent.find_matched_controls = AsyncMock(return_value=None)
        
        result = await agent.run()
        
        # Should fallback to query_as_control
        assert mock_gcs.submit_inference_job.call_count == 1
        assert result.control_strategy_effective == ControlStrategy.QUERY_AS_CONTROL
```

---

## Phase 3: Frontend Tests

### Task 3.1: Set Up Jest

- [ ] **3.1.1** Create `frontend/jest.config.js`:

```javascript
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  testPathIgnorePatterns: ['/node_modules/', '/.next/'],
};
```

- [ ] **3.1.2** Create `frontend/jest.setup.js`:

```javascript
import '@testing-library/jest-dom';
```

---

### Task 3.2: Component Tests

- [ ] **3.2.1** Create `frontend/__tests__/components/RunForm.test.tsx`:

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RunForm } from '@/components/runs/RunForm';

describe('RunForm', () => {
  const queryClient = new QueryClient();
  
  it('disables submit button when query is too short', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    const button = screen.getByRole('button', { name: /start analysis/i });
    expect(button).toBeDisabled();
  });
  
  it('enables submit button when query is valid', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'How would lung fibroblasts respond to TGF-beta?' } });
    
    const button = screen.getByRole('button', { name: /start analysis/i });
    expect(button).toBeEnabled();
  });
});
```

---

## Phase 4: Integration Tests

### Task 4.1: End-to-End Workflow Tests

- [ ] **4.1.1** Create `tests/integration/test_workflow.py`:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_synthetic_control(test_client, test_database):
    # Create run
    response = await test_client.post("/api/v1/runs", json={
        "query": "How would T cells respond to IL-6?",
        "control_strategy": "synthetic_control",
    })
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    
    # Wait for completion (with timeout)
    for _ in range(60):
        status = await test_client.get(f"/api/v1/runs/{run_id}")
        if status.json()["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(10)
    
    # Verify result
    result = await test_client.get(f"/api/v1/runs/{run_id}/result")
    assert result.status_code == 200
    assert result.json()["control_strategy_effective"] == "synthetic_control"
```

---

## Phase 5: Load Testing

### Task 5.1: Performance Tests

- [ ] **5.1.1** Test concurrent run creation
- [ ] **5.1.2** Test vector search latency with 10M cells
- [ ] **5.1.3** Test database connection pool under load
- [ ] **5.1.4** Document performance benchmarks

---

## Definition of Done

- [ ] >80% code coverage for backend
- [ ] All API endpoints tested
- [ ] All retrieval strategies tested
- [ ] Ontology resolution tested with fallbacks
- [ ] Control strategy tests pass
- [ ] Frontend component tests pass
- [ ] Integration tests pass in CI
- [ ] Performance benchmarks documented

---

## Next Sprint

**Sprint 12: Documentation & Launch Prep** - Final documentation and launch preparation.
