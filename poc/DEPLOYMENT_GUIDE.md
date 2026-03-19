# Hosted Agent Deployment to Microsoft Foundry

End-to-end guide for deploying an Agent Framework agent to Foundry using ACR build and the Python SDK.

## Prerequisites

- Azure CLI (`az`) v2.80+
- Python 3.10+
- An Azure AI Foundry project with a deployed chat model (e.g., `gpt-4.1`)
- An Azure Container Registry (ACR)

## 1. Project Setup

The `poc/` folder contains:

| File | Purpose |
|---|---|
| `main.py` | Agent code (Seattle Hotel Agent sample from foundry-samples) |
| `Dockerfile` | Python 3.12-slim container, exposes port 8088 |
| `.dockerignore` | Excludes .env, venvs, pycache, etc. |
| `requirements.txt` | Agent framework SDK + dependencies |
| `agent.yaml` | Agent definition for `azd` workflows |
| `deploy_agent.py` | Python SDK deployment script |

## 2. Test Locally

```powershell
cd poc

$env:PROJECT_ENDPOINT = "https://<your-resource>.services.ai.azure.com/api/projects/<your-project>"
$env:MODEL_DEPLOYMENT_NAME = "gpt-4.1"

python main.py
```

In a second terminal:

```powershell
$body = @{
    input = "Find me hotels in Seattle from 2026-04-01 to 2026-04-04 under 200 per night"
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8088/responses -Method Post -Body $body -ContentType "application/json"
```

## 3. Build & Push Image via ACR Build

From the `poc/` directory:

```powershell
az acr build --registry <ACR_NAME> --image <IMAGE_NAME>:<TAG> --platform linux/amd64 .
```

Example:

```powershell
az acr build --registry aicouncilacr285 --image seattle-hotel-agent:v1 --platform linux/amd64 .
```

> **Note:** Always use `--platform linux/amd64`. ARM64 images (e.g., from Apple Silicon) are not compatible.

## 4. Create Capability Host (One-Time Per Foundry Account)

The capability host is **per Foundry account** (not per project or resource group). All projects under the account share it.

```powershell
az rest --method put `
    --url "https://management.azure.com/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/capabilityHosts/accountcaphost?api-version=2025-10-01-preview" `
    --headers "content-type=application/json" `
    --body '{\"properties\":{\"capabilityHostKind\":\"Agents\",\"enablePublicHostingEnvironment\":true}}'
```

Verify it succeeded:

```powershell
az rest --method get `
    --url "https://management.azure.com/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/capabilityHosts/accountcaphost?api-version=2025-10-01-preview" `
    -o json
```

Look for `"provisioningState": "Succeeded"`.

## 5. Configure ACR Access (CRITICAL)

> **This is the step most likely to be missed.** Without it, the agent version will be created but the deployment will fail to start silently — no error, just no response.

You need **three things**:

### 5a. Get the Project's Managed Identity

The **project** has its own system-assigned managed identity, separate from the account's identity. Get it with:

```powershell
az rest --method get `
    --url "https://management.azure.com/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/projects/<PROJECT_NAME>?api-version=2025-04-01-preview" `
    --query "identity.principalId" -o tsv
```

### 5b. Grant AcrPull to the Project Identity

```powershell
az role assignment create `
    --assignee "<PROJECT_PRINCIPAL_ID>" `
    --role "AcrPull" `
    --scope "/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.ContainerRegistry/registries/<ACR_NAME>"
```

### 5c. Create the ACR Connection on the Project

This tells the Foundry platform which registry to use and how to authenticate:

```powershell
az rest --method put `
    --url "https://management.azure.com/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/projects/<PROJECT_NAME>/connections/acr-connection?api-version=2025-04-01-preview" `
    --headers "content-type=application/json" `
    --body '{\"properties\":{\"category\":\"ContainerRegistry\",\"target\":\"<ACR_NAME>.azurecr.io\",\"authType\":\"ManagedIdentity\",\"isSharedToAll\":true,\"credentials\":{\"clientId\":\"<PROJECT_PRINCIPAL_ID>\",\"resourceId\":\"/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.ContainerRegistry/registries/<ACR_NAME>\"},\"metadata\":{\"ResourceId\":\"/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.ContainerRegistry/registries/<ACR_NAME>\"}}}'
```

### 5d. Grant Cognitive Services Contributor to the Project Identity

So the hosted agent's container can call the deployed model:

```powershell
az role assignment create `
    --assignee "<PROJECT_PRINCIPAL_ID>" `
    --role "53ca6127-db72-4b80-b1b0-d745d6d5456d" `
    --scope "/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>"
```

## 6. Deploy the Agent

Update `IMAGE_TAG` in `deploy_agent.py` to match the tag from step 3, then:

```powershell
cd poc
python deploy_agent.py
```

This calls `project.agents.create_version()` to register the hosted agent.

## 7. Start the Deployment

The deployment should auto-start. If not, use:

```powershell
az cognitiveservices agent start `
    --account-name <ACCOUNT_NAME> `
    --project-name <PROJECT_NAME> `
    --name <AGENT_NAME> `
    --agent-version 1
```

## 8. Verify

```powershell
az cognitiveservices agent show `
    --account-name <ACCOUNT_NAME> `
    --project-name <PROJECT_NAME> `
    --name <AGENT_NAME> `
    -o json
```

## 9. Invoke the Deployed Agent

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

project_client = AIProjectClient(
    endpoint="https://<ACCOUNT_NAME>.services.ai.azure.com/api/projects/<PROJECT_NAME>",
    credential=DefaultAzureCredential(),
)

openai_client = project_client.get_openai_client()

response = openai_client.responses.create(
    input=[{"role": "user", "content": "Find hotels in Seattle for April 1-4 under $200/night"}],
    extra_body={"agent_reference": {"name": "<AGENT_NAME>", "version": "1", "type": "agent_reference"}},
)

print(response.output_text)
```

## 10. Cleanup

Stop the deployment:

```powershell
az cognitiveservices agent stop `
    --account-name <ACCOUNT_NAME> `
    --project-name <PROJECT_NAME> `
    --name <AGENT_NAME> `
    --agent-version 1
```

Delete the agent version:

```python
project.agents.delete_version(agent_name="<AGENT_NAME>", agent_version="1")
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent created but no response | Missing ACR connection on project | Follow Step 5c |
| Agent created but no response | AcrPull on wrong identity (account vs project) | Grant AcrPull to the **project's** principalId (Step 5b) |
| `InvalidAcrPullCredentials` | Managed identity can't auth to ACR | Check role assignment + connection credentials |
| `AcrImageNotFound` | Wrong image name/tag | Verify image exists: `az acr repository show-tags --name <ACR> --repository <IMAGE>` |
| Deployment stuck in Creating | Capability host not ready | Check provisioningState (Step 4) |
