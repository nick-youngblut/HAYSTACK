# Biological Database Integration

### 7.1 External API Configuration

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx


class BiologicalDatabaseClient:
    """Async client for biological database APIs."""
    
    def __init__(self, config: DatabaseAPIConfig):
        """
        Initialize database client.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AsyncRateLimiter(config.requests_per_minute)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
    )
    async def query_kegg(self, endpoint: str, params: dict) -> dict:
        """Query KEGG API with retry logic."""
        async with self._rate_limiter:
            response = await self.client.get(
                f"https://rest.kegg.jp/{endpoint}",
                params=params,
            )
            response.raise_for_status()
            return self._parse_kegg_response(response.text)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
    )
    async def query_reactome(self, endpoint: str, params: dict) -> dict:
        """Query Reactome API with retry logic."""
        async with self._rate_limiter:
            response = await self.client.get(
                f"https://reactome.org/ContentService/{endpoint}",
                params=params,
            )
            response.raise_for_status()
            return response.json()

    # Literature search methods
    async def search_pubmed(self, query: str, max_results: int = 20) -> list[dict]:
        """Search PubMed via E-utilities API."""
        ...
    
    async def search_semantic_scholar(self, query: str, max_results: int = 20) -> list[dict]:
        """Search Semantic Scholar Graph API."""
        ...
    
    async def search_biorxiv(self, query: str, max_results: int = 20) -> list[dict]:
        """Search bioRxiv/medRxiv API."""
        ...
    
    async def acquire_full_text(self, doi: str) -> dict:
        """Acquire full-text PDF and convert to markdown."""
        ...
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

### 7.2 Supported Databases

| Database | Purpose | Endpoint |
|----------|---------|----------|
| KEGG | Pathways, drug targets | https://rest.kegg.jp |
| Reactome | Pathway analysis | https://reactome.org/ContentService |
| UniProt | Protein information | https://rest.uniprot.org |
| PubMed | Literature search | https://eutils.ncbi.nlm.nih.gov |
| Semantic Scholar | Literature search | https://api.semanticscholar.org |
| bioRxiv/medRxiv | Preprint search | https://api.biorxiv.org |
| CORE | Open access full text | https://api.core.ac.uk |
| Europe PMC | Literature/full text | https://www.ebi.ac.uk/europepmc |
| Unpaywall | OA availability | https://api.unpaywall.org |
| Cell Ontology | Cell type hierarchy | Local OBO file |
| Gene Ontology | GO terms | https://api.geneontology.org |

---

## Related Specs

- `specification/tools.md`
- `specification/data-models.md`
- `specification/database.md`
