# Vanilla Planner Agent — Hosted Agent Deployment to Microsoft Foundry

Custom (no-framework) Planner Agent using `FoundryCBAgent`. Same ReAct loop as `planner_agent.py`, wrapped in a thin HTTP adapter for the Foundry Responses API.

## Key Difference from `planner_poc/`

| | `planner_poc/` | `vanilla_planner_poc/` |
|---|---|---|
| Base | Agent Framework SDK (`Agent` + `AzureAIAgentClient`) | Custom `FoundryCBAgent` (no framework) |
| Tool calling | Framework handles function-calling dispatch | Agent implements its own ReAct loop internally |
| LLM calls | Framework manages `responses.create()` | Agent calls `chat.completions.create()` itself |
| Flexibility | Constrained to framework patterns | Full control over agent logic |

## Prerequisites

- Azure CLI (`az`) v2.80+
- Python 3.10+
- An Azure AI Foundry project with a deployed chat model (e.g., `gpt-4.1`)
- An Azure Container Registry (ACR)
- A Cosmos DB Gremlin account with `indigokg/knowledgegraph`

## 1. Project Files

| File | Purpose |
|---|---|
| `main.py` | PlannerCBAgent (FoundryCBAgent wrapper) + PlannerAgent (ReAct loop) + GraphOntologyTool |
| `cosmos_helpers.py` | Cosmos DB Gremlin connection & query helpers |
| `Dockerfile` | Python 3.12-slim container, exposes port 8088 |
| `.dockerignore` | Excludes .env, venvs, pycache, etc. |
| `requirements.txt` | agentserver-core SDK + openai + Gremlin dependencies |
| `agent.yaml` | Agent definition for `azd` workflows |
| `deploy_agent.py` | Python SDK deployment script |

## 2. Test Locally

```powershell
cd vanilla_planner_poc
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt

$env:PROJECT_ENDPOINT = "https://new-agent-host-resource.services.ai.azure.com/api/projects/new-agent-host"
$env:MODEL_DEPLOYMENT_NAME = "gpt-4.1"
$env:COSMOS_DB_ENDPOINT = "cosmosdb-gremlin-abpatra.gremlin.cosmos.azure.com"
$env:COSMOS_DB_KEY = "<your-cosmos-key>"

python main.py
```

In a second terminal:

```powershell
$body = @{
    input = "Which data sources should I check for flight delay analysis?"
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8088/responses -Method Post -Body $body -ContentType "application/json"
```

## 3. Build & Push Image

```powershell
cd vanilla_planner_poc
az acr build --registry aicouncilacr285 --image vanilla-planner-agent:v1 --platform linux/amd64 .
```

## 4. Deploy the Agent

```powershell
$env:COSMOS_DB_KEY = "<your-cosmos-key>"
python deploy_agent.py
```

## 5. Verify

```powershell
az cognitiveservices agent show --account-name new-agent-host-resource --project-name new-agent-host --name vanilla-planner-agent -o json
```

## 6. Start (if not auto-started)

```powershell
az cognitiveservices agent start --account-name new-agent-host-resource --project-name new-agent-host --name vanilla-planner-agent --agent-version 1
```

## 7. Invoke

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

project_client = AIProjectClient(
    endpoint="https://new-agent-host-resource.services.ai.azure.com/api/projects/new-agent-host",
    credential=DefaultAzureCredential(),
)

openai_client = project_client.get_openai_client()

response = openai_client.responses.create(
    input=[{"role": "user", "content": "Which data sources should I check for flight delay analysis?"}],
    extra_body={"agent_reference": {"name": "vanilla-planner-agent", "version": "1", "type": "agent_reference"}},
)

print(response.output_text)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent created but no response | Missing ACR connection | See planner_poc DEPLOYMENT_GUIDE Step 5 |
| Gremlin connection error | Missing Cosmos DB env vars | Check `COSMOS_DB_ENDPOINT` and `COSMOS_DB_KEY` in deploy_agent.py |
| `AcrImageNotFound` | Wrong image name/tag | `az acr repository show-tags --name aicouncilacr285 --repository vanilla-planner-agent` |
