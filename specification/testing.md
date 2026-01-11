# Testing Strategy

### 15.1 Backend Tests

```python
# backend/tests/conftest.py

import pytest
import asyncio
from httpx import AsyncClient
from backend.main import app
from backend.services.database import database


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def mock_database(mocker):
    """Mock database for unit tests."""
    mock_db = mocker.MagicMock()
    mocker.patch("backend.services.database.database", mock_db)
    return mock_db
```

```python
# backend/tests/test_api/test_runs.py

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_run(client: AsyncClient, mock_database):
    """Test creating a new run."""
    response = await client.post(
        "/api/v1/runs/",
        json={"query": "How would lung fibroblasts respond to TGF-beta?"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_get_run_status(client: AsyncClient, mock_database):
    """Test getting run status."""
    mock_database.get_run.return_value = {
        "run_id": "test_123",
        "status": "running",
        "iterations": [],
        "config": {"max_iterations": 5},
    }
    
    response = await client.get("/api/v1/runs/test_123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test_123"
    assert data["status"] == "running"
```

### 15.2 Frontend Tests

```typescript
// frontend/__tests__/components/RunForm.test.tsx

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RunForm } from "@/components/runs/RunForm";

const queryClient = new QueryClient();

describe("RunForm", () => {
  it("renders query input", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    expect(screen.getByPlaceholderText(/lung fibroblasts/i)).toBeInTheDocument();
  });
  
  it("disables submit for short queries", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    const button = screen.getByRole("button", { name: /start analysis/i });
    expect(button).toBeDisabled();
  });
});
```

### 15.3 Agent + Tool Tests

Test tools as pure functions with mocked runtime/context, then add agent-level integration tests that exercise tool selection and orchestration. Separate fast unit tests from optional LLM-backed integration tests.

```python
# backend/tests/agents/tools/test_query_tool.py

import pytest
from backend.agents.tools.query_tool import query_tool


class MockRuntime:
    """Minimal ToolRuntime stand-in for unit tests."""

    def __init__(self, config=None):
        self.config = config or {}


def test_query_tool_returns_ranked_items(mocker):
    runtime = MockRuntime(config={"collections": {"papers": "test_papers"}})
    mocker.patch("backend.agents.tools.query_tool.search_index", return_value=[
        {"id": "p1", "score": 0.91},
        {"id": "p2", "score": 0.84},
    ])

    result = query_tool("tgf-beta fibroblasts", runtime=runtime)

    assert result["items"][0]["id"] == "p1"
    assert result["items"][0]["score"] >= result["items"][1]["score"]
```

```python
# backend/tests/agents/test_agent_integration.py

import pytest
from langchain.chat_models import init_chat_model
from backend.agents.main import build_agent


@pytest.mark.asyncio
async def test_agent_routes_to_tool(mocker):
    # Mock tool execution to make the agent deterministic
    mocker.patch("backend.agents.tools.query_tool.query_tool", return_value={
        "items": [{"id": "p1", "score": 0.91}],
    })
    model = init_chat_model("openai:gpt-5-mini", temperature=0)
    agent = build_agent(model=model)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Find TGF-beta fibroblast papers"}]})

    assert "p1" in str(result)
```

Optional integration tests should be marked (e.g., `@pytest.mark.integration`) and run with real model keys for end-to-end validation.

### 15.4 Cell Ontology Integration Tests

```python
# tests/unit/test_query_understanding.py

import pytest
from unittest.mock import AsyncMock, patch

from haystack.orchestrator.agents.query_understanding import (
    resolve_cell_type_with_fallback,
)


class TestCellTypeResolution:
    """Tests for cell type resolution in Query Understanding agent."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_success(self):
        """Test successful resolution via semantic search."""
        with patch("haystack.orchestrator.services.ontology.CellOntologyService") as mock:
            mock.get_instance.return_value.semantic_search = AsyncMock(
                return_value={
                    "fibroblast": [
                        {"term_id": "CL:0000057", "name": "fibroblast", "distance": 0.05}
                    ]
                }
            )
            
            result = await resolve_cell_type_with_fallback("fibroblast")
            
            assert result["resolved"] is True
            assert result["term_id"] == "CL:0000057"
            assert result["method"] == "semantic"
    
    @pytest.mark.asyncio
    async def test_ols_fallback(self):
        """Test fallback to OLS when semantic search fails."""
        with patch("haystack.orchestrator.services.ontology.CellOntologyService") as mock:
            mock.get_instance.return_value.semantic_search = AsyncMock(
                return_value={"rare cell": "No ontology ID found"}
            )
            mock.get_instance.return_value.query_ols = AsyncMock(
                return_value={
                    "rare cell": [
                        {"term_id": "CL:0001234", "name": "rare cell type"}
                    ]
                }
            )
            
            result = await resolve_cell_type_with_fallback("rare cell")
            
            assert result["resolved"] is True
            assert result["method"] == "ols"
    
    @pytest.mark.asyncio
    async def test_unresolved_cell_type(self):
        """Test handling of unresolvable cell type."""
        with patch("haystack.orchestrator.services.ontology.CellOntologyService") as mock:
            mock.get_instance.return_value.semantic_search = AsyncMock(
                return_value={"unknown": "No ontology ID found"}
            )
            mock.get_instance.return_value.query_ols = AsyncMock(
                return_value={"unknown": []}
            )
            
            result = await resolve_cell_type_with_fallback("unknown")
            
            assert result["resolved"] is False
            assert "warning" in result
```

### 15.5 Ontology-Guided Strategy Tests

```python
# tests/unit/test_ontology_guided_strategy.py

import pytest
from unittest.mock import AsyncMock, patch

from haystack.orchestrator.strategies.ontology_guided import OntologyGuidedStrategy
from haystack.shared.models.queries import StructuredQuery, ICLTaskType


class TestOntologyGuidedStrategy:
    """Tests for ontology-guided retrieval strategy."""
    
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()
    
    @pytest.fixture
    def strategy(self, mock_db):
        with patch("haystack.orchestrator.services.ontology.CellOntologyService"):
            return OntologyGuidedStrategy(mock_db)
    
    @pytest.mark.asyncio
    async def test_returns_empty_without_cl_id(self, strategy):
        """Strategy returns empty if no cell type CL ID."""
        query = StructuredQuery(
            raw_query="test",
            task_type=ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
            cell_type_query="fibroblast",
            cell_type_cl_id=None,  # No CL ID
        )
        
        result = await strategy.retrieve(query)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_uses_parent_types_first(self, strategy, mock_db):
        """Strategy prioritizes parent types over children."""
        with patch.object(strategy.ontology_service, "get_neighbors") as mock_neighbors:
            mock_neighbors.return_value = {
                "CL:0000057": [
                    {"term_id": "CL:0000548", "name": "animal cell", "relationship_type": "is_a"},
                    {"term_id": "CL:0002553", "name": "fibroblast of lung", "relationship_type": "is_a_inverse"},
                ]
            }
            
            mock_db.execute_query.return_value = [
                {"group_id": "g1", "dataset": "parse_pbmc", "cell_type_cl_id": "CL:0000548",
                 "cell_type_name": "animal cell", "perturbation_name": "TGF-beta", "n_cells": 100}
            ]
            
            query = StructuredQuery(
                raw_query="test",
                task_type=ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
                cell_type_query="fibroblast",
                cell_type_cl_id="CL:0000057",
                perturbation_resolved="TGF-beta",
            )
            
            result = await strategy.retrieve(query, max_results=10)
            
            assert len(result) > 0
            # First result should be from parent type (is_a)
            assert "is_a" in result[0].rationale
```

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
- `specification/frontend.md`
- `specification/agents.md`
- `specification/tools.md`
