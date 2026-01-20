 # Sprint 10: Deployment & Configuration

**Duration**: 1 week  
**Dependencies**: Sprint 09 (Docker Containers)  
**Goal**: Deploy to GCP and configure production environment.

---

## Overview

> **Spec Reference**: `./specification/deployment.md`, `./specification/configuration.md`

This sprint implements:
- Cloud Run deployment
- Dynaconf configuration management
- Secret Manager integration
- Monitoring and logging setup

---

## Phase 1: Cloud Run Deployment

### Task 1.1: Deploy API Container

> **Spec Reference**: `./specification/deployment.md` (Section 12.5)

- [ ] **1.1.1** Deploy Cloud Run service:

```bash
gcloud run deploy haystack-api \
  --image=us-east1-docker.pkg.dev/PROJECT_ID/haystack/api:latest \
  --region=us-east1 \
  --platform=managed \
  --min-instances=0 \
  --max-instances=10 \
  --memory=4Gi \
  --cpu=2 \
  --timeout=60s \
  --service-account=haystack-cloudrun@PROJECT_ID.iam.gserviceaccount.com \
  --vpc-connector=haystack-connector \
  --set-secrets="DATABASE_PASSWORD=haystack-db-password:latest,SENDGRID_API_KEY=sendgrid-api-key:latest" \
  --set-env-vars="HAYSTACK_ENV=production,GCP_PROJECT_ID=PROJECT_ID"
```

- [ ] **1.1.2** Configure custom domain (optional)
- [ ] **1.1.3** Enable IAP authentication:

```bash
gcloud iap web enable \
  --resource-type=cloud-run \
  --service=haystack-api \
  --region=us-east1
```

- [ ] **1.1.4** Configure IAP OAuth consent screen
- [ ] **1.1.5** Add authorized users/groups

---

### Task 1.2: Verify Deployment

- [ ] **1.2.1** Test health endpoint
- [ ] **1.2.2** Test API endpoints with authentication
- [ ] **1.2.3** Verify frontend loads correctly
- [ ] **1.2.4** Test run creation end-to-end

---

## Phase 2: Configuration Management

### Task 2.1: Set Up Dynaconf

> **Spec Reference**: `./specification/configuration.md`

- [ ] **2.1.1** Create `shared/config.py`:

```python
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="HAYSTACK",
    settings_files=["settings.yaml", ".secrets.yaml"],
    environments=True,
    env_switcher="HAYSTACK_ENV",
    load_dotenv=True,
)
```

- [ ] **2.1.2** Create `settings.yaml`:

```yaml
default:
  # App settings
  app_name: HAYSTACK
  app_url: http://localhost:8080
  
  # Database
  database:
    host: localhost
    port: 5432
    name: haystack
    user: haystack_app
    pool_size: 5
    max_overflow: 10
  
  # GCS
  gcs:
    project_id: PROJECT_ID
    atlases_bucket: haystack-atlases
    models_bucket: haystack-models
    batch_io_bucket: haystack-batch-io
    results_bucket: haystack-results
  
  # Batch
  batch:
    region: us-east1
    orchestrator_machine_type: e2-standard-4
    inference_machine_type: a2-highgpu-1g
    job_timeout_seconds: 3600
  
  # LLM
  llm:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096
  
  # Iteration
  iteration:
    max_iterations: 5
    score_threshold: 7
    plateau_window: 3
  
  # External APIs
  external_apis:
    max_retries: 3
    base_delay_seconds: 1.0
    requests_per_minute: 30

development:
  debug: true
  database:
    host: localhost
    password: devpassword
  
production:
  debug: false
  app_url: https://haystack.example.com
  database:
    instance_connection_name: PROJECT_ID:us-east1:haystack-db
```

---

### Task 2.2: Configure Secret Manager

- [ ] **2.2.1** Create secrets:

```bash
# Database password
echo -n "secure_password" | gcloud secrets create haystack-db-password \
  --data-file=- --replication-policy="automatic"

# SendGrid API key
echo -n "SG.xxx" | gcloud secrets create sendgrid-api-key \
  --data-file=- --replication-policy="automatic"

# OpenAI API key
echo -n "sk-xxx" | gcloud secrets create openai-api-key \
  --data-file=- --replication-policy="automatic"

# Anthropic API key
echo -n "sk-ant-xxx" | gcloud secrets create anthropic-api-key \
  --data-file=- --replication-policy="automatic"
```

- [ ] **2.2.2** Grant secret access to service accounts:

```bash
for SA in haystack-cloudrun haystack-orchestrator; do
  for SECRET in haystack-db-password sendgrid-api-key openai-api-key anthropic-api-key; do
    gcloud secrets add-iam-policy-binding $SECRET \
      --member="serviceAccount:$SA@PROJECT_ID.iam.gserviceaccount.com" \
      --role="roles/secretmanager.secretAccessor"
  done
done
```

---

## Phase 3: Monitoring & Logging

### Task 3.1: Configure Cloud Logging

- [ ] **3.1.1** Set up structured logging in application:

```python
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

def setup_logging():
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, logging.StreamHandler()],
    )
```

- [ ] **3.1.2** Add run_id to all log entries:

```python
class RunContextFilter(logging.Filter):
    def __init__(self, run_id: str):
        self.run_id = run_id
    
    def filter(self, record):
        record.run_id = self.run_id
        return True
```

- [ ] **3.1.3** Create log-based metrics for key events

---

### Task 3.2: Configure Cloud Monitoring

- [ ] **3.2.1** Create uptime checks:

```bash
gcloud monitoring uptime-check-configs create haystack-api-uptime \
  --display-name="HAYSTACK API Health Check" \
  --http-check-path="/api/v1/health/" \
  --monitored-resource-type="cloud_run_revision"
```

- [ ] **3.2.2** Create alert policies:
  - API error rate > 5%
  - Batch job failure rate > 10%
  - Response latency P95 > 5s

- [ ] **3.2.3** Set up notification channels (email, Slack)

---

### Task 3.3: Create Monitoring Dashboard

- [ ] **3.3.1** Create dashboard with:
  - Run success/failure rates
  - Average run duration by phase
  - Grounding score distribution
  - API request latency
  - Batch job queue depth

---

## Phase 4: Backup & Recovery

### Task 4.1: Configure Database Backups

- [ ] **4.1.1** Verify automated backups enabled:

```bash
gcloud sql instances patch haystack-db \
  --backup-start-time=03:00 \
  --enable-point-in-time-recovery
```

- [ ] **4.1.2** Set backup retention (7 days default)
- [ ] **4.1.3** Document recovery procedure

---

### Task 4.2: Configure GCS Versioning

- [ ] **4.2.1** Enable versioning on models bucket:

```bash
gsutil versioning set on gs://haystack-models
```

- [ ] **4.2.2** Document rollback procedure for model updates

---

## Definition of Done

- [ ] Cloud Run service deployed and accessible
- [ ] IAP authentication working
- [ ] All secrets in Secret Manager
- [ ] Dynaconf configuration working
- [ ] Structured logging enabled
- [ ] Monitoring dashboard created
- [ ] Backup strategy documented

---

## Next Sprint

**Sprint 11: Testing** - Implement comprehensive test suite.
