# Configuration

### 11.1 Backend Configuration (Dynaconf)

```yaml
# backend/settings.yml

default:
  app_name: "HAYSTACK"
  version: "1.0.0"
  debug: false
  log_level: "INFO"
  
  # LLM defaults
  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-5-20250929"
    temperature: 0.7
    max_tokens: 4096
  
  # Iteration defaults
  iteration:
    max_iterations: 5
    score_threshold: 7
    plateau_window: 3
    min_improvement: 1
  
  # Database API settings
  database_apis:
    max_retries: 3
    base_delay_seconds: 1.0
    max_delay_seconds: 30.0
    requests_per_minute: 30

  # Literature search settings
  literature:
    max_results_per_database: 20
    default_databases:
      - pubmed
      - semantic_scholar
      - biorxiv
    enable_full_text: true
    max_full_text_chars: 100_000
    enable_image_descriptions: false
    requests_per_minute: 30
    request_timeout_seconds: 30
    max_retries: 3
    cache_ttl_seconds: 3600
    cache_max_size: 100
    biorxiv_date_window_days: 365
    include_medrxiv: true

  # Ontology configuration
  ontology:
    cell:
      enabled: true
      embedding_model: "text-embedding-3-small"
      embedding_dimension: 1536
      default_k: 3
      default_distance_threshold: 0.7
    
    ols:
      base_url: "https://www.ebi.ac.uk/ols4/api"
      request_timeout: 30
      max_concurrent_requests: 5
      retry_attempts: 3
    
    gcs:
      bucket: "haystack-data"
      prefix: "ontology"
  
  # GCP Batch configuration for STACK inference
  batch:
    region: "us-east1"
  # GCP Batch configuration
  batch:
    region: "us-east1"
    
    # CPU Orchestrator job (runs the agent workflow)
    orchestrator:
      machine_type: "e2-standard-4"  # 4 vCPU, 16 GB RAM
      job_timeout_seconds: 7200      # 2 hours max
      container_image: "us-east1-docker.pkg.dev/arc-prod/haystack/orchestrator:latest"
      service_account: "haystack-orchestrator-sa@arc-prod.iam.gserviceaccount.com"
    
    # GPU Inference job (STACK model)
    inference:
      machine_type: "a2-highgpu-1g"  # NVIDIA A100 80GB
      accelerator_type: "nvidia-tesla-a100"
      accelerator_count: 1
      boot_disk_size_gb: 200
      job_timeout_seconds: 1800      # 30 minutes max
      poll_interval_seconds: 10
      container_image: "us-east1-docker.pkg.dev/arc-prod/haystack/inference:latest"
      service_account: "haystack-inference-sa@arc-prod.iam.gserviceaccount.com"
  
  # Email notifications
  email:
    enabled: true
    from_address: "haystack@arc.institute"
    from_name: "HAYSTACK"

dev:
  debug: true
  log_level: "DEBUG"

  ontology:
    cell:
      default_distance_threshold: 0.8  # More lenient in dev
  
  email:
    enabled: false  # Disable emails in development
  
  database:
    instance_connection_name: "arc-dev:us-east1:haystack-dev"
    database_name: "haystack"
    user: "haystack_app"
  
  gcs:
    project_id: "arc-dev"
    bucket_name: "haystack-dev"
    atlases_prefix: "atlases/"
    stack_model_prefix: "models/stack/"
    results_prefix: "results/"
    batch_io_prefix: "batch-io/"
  
  batch:
    orchestrator:
      container_image: "us-east1-docker.pkg.dev/arc-dev/haystack/orchestrator:latest"
      service_account: "haystack-orchestrator-sa@arc-dev.iam.gserviceaccount.com"
    inference:
      container_image: "us-east1-docker.pkg.dev/arc-dev/haystack/inference:latest"
      service_account: "haystack-inference-sa@arc-dev.iam.gserviceaccount.com"

prod:
  database:
    instance_connection_name: "arc-prod:us-east1:haystack-prod"
    database_name: "haystack"
    user: "haystack_app"

  ontology:
    ols:
      max_concurrent_requests: 10
  
  gcs:
    project_id: "arc-prod"
    bucket_name: "haystack-prod"
    atlases_prefix: "atlases/"
    stack_model_prefix: "models/stack/"
    results_prefix: "results/"
    batch_io_prefix: "batch-io/"
```

### 11.2 Environment Variables

```bash
# .env.example

# Environment selection
DYNACONF=dev  # or prod

# LLM API Keys (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Email notifications (SendGrid)
SENDGRID_API_KEY=SG...

# Literature API credentials
CORE_API_KEY=
UNPAYWALL_EMAIL=

# Database (Cloud SQL)
# Connection handled via Cloud SQL Connector with IAM auth

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=arc-dev

# Frontend
FRONTEND_OUT_DIR=/app/frontend/out
NEXT_PUBLIC_API_URL=https://haystack.arc.institute/api

# Application
PORT=8080
```

---

## Related Specs

- `specification/deployment.md`
- `specification/agents.md`
