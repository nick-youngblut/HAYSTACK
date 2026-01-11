# Dependencies

### 16.1 Backend Dependencies

```toml
# backend/pyproject.toml

[project]
name = "haystack"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    # Web framework
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    
    # Validation & settings
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "dynaconf>=3.2.4",
    
    # Database
    "asyncpg>=0.29.0",
    "pgvector>=0.2.0",
    "cloud-sql-python-connector[asyncpg]>=1.6.0",
    
    # Google Cloud
    "google-cloud-storage>=2.14.0",
    "google-cloud-secret-manager>=2.18.0",
    "google-cloud-batch>=0.17.0",
    
    # Email notifications
    "sendgrid>=6.11.0",
    
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
    "docling>=2.60.0",
    "lxml>=5.0.0",
    
    # API clients
    "httpx>=0.27.0",
    "tenacity>=8.0.0",
    
    # Biological analysis
    "gseapy>=1.0.0",
    "networkx>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]
```

### 16.2 Frontend Dependencies

```json
{
  "name": "haystack-frontend",
  "version": "1.0.0",
  "dependencies": {
    "@tanstack/react-query": "^5.90.10",
    "@headlessui/react": "^2.2.9",
    "@heroicons/react": "^2.2.0",
    "axios": "^1.13.2",
    "clsx": "^2.1.1",
    "date-fns": "^4.1.0",
    "next": "16.0.3",
    "react": "19.2.0",
    "react-dom": "19.2.0",
    "react-markdown": "^9.0.0",
    "tailwind-merge": "^3.4.0",
    "zod": "^4.1.12",
    "zustand": "^4.4.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4",
    "@testing-library/jest-dom": "^6.8.0",
    "@testing-library/react": "^16.3.0",
    "@types/node": "^20",
    "@types/react": "^19",
    "eslint": "^9",
    "eslint-config-next": "16.0.3",
    "jest": "^30.2.0",
    "jest-environment-jsdom": "^30.2.0",
    "tailwindcss": "^4",
    "typescript": "^5"
  }
}
```

---

## Related Specs

- `specification/deployment.md`
