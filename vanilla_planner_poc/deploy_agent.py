"""
Deploy the Vanilla Planner Agent to Azure AI Foundry as a hosted agent.

Prerequisites:
  Build & push the image manually first (from the vanilla_planner_poc/ directory):

    az acr build --registry <ACR_NAME> --image vanilla-planner-agent:<TAG> --platform linux/amd64 .

  Then set the IMAGE_TAG below to match, and run:

    python deploy_agent.py

"""

import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    HostedAgentDefinition,
    ProtocolVersionRecord,
    AgentProtocol,
)
from azure.core.pipeline.policies import CustomHookPolicy
from azure.identity import DefaultAzureCredential

# ── Configuration ───────────────────────────────────────────────────────────
PROJECT_ENDPOINT = "https://new-agent-host-resource.services.ai.azure.com/api/projects/new-agent-host"
ACR_NAME = "aicouncilacr285"                      # Azure Container Registry name (no .azurecr.io)
IMAGE_NAME = "vanilla-planner-agent"               # Repository name inside ACR
IMAGE_TAG = "v2"                                   # Set this to the tag you used in az acr build
AGENT_NAME = "vanilla-planner-agent"               # Agent name in Foundry
MODEL_DEPLOYMENT_NAME = "gpt-4.1"                  # Model deployment to use

# Cosmos DB — these are passed as environment variables to the container
COSMOS_DB_ENDPOINT = "cosmosdb-gremlin-abpatra.gremlin.cosmos.azure.com"
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY", "")  # Set via env var before running

FULL_IMAGE = f"{ACR_NAME}.azurecr.io/{IMAGE_NAME}:{IMAGE_TAG}"


# ── Deploy to Foundry ──────────────────────────────────────────────────────
def deploy_agent():
    """Create a hosted agent version in Foundry pointing at the new image."""

    if not COSMOS_DB_KEY:
        print("WARNING: COSMOS_DB_KEY is empty. Set it as an environment variable.")

    def _add_preview_header(request):
        request.http_request.headers["Foundry-Features"] = "HostedAgents=V1Preview"

    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
        allow_preview=True,
        per_call_policies=[CustomHookPolicy(raw_request_hook=_add_preview_header)],
    )

    print(f"Deploying image: {FULL_IMAGE}")

    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=HostedAgentDefinition(
            container_protocol_versions=[
                ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="v2")
            ],
            cpu="1",
            memory="2Gi",
            image=FULL_IMAGE,
            environment_variables={
                "PROJECT_ENDPOINT": PROJECT_ENDPOINT,
                "MODEL_DEPLOYMENT_NAME": MODEL_DEPLOYMENT_NAME,
                "COSMOS_DB_ENDPOINT": COSMOS_DB_ENDPOINT,
                "COSMOS_DB_KEY": COSMOS_DB_KEY,
            },
        ),
    )

    print(f"Agent created: {agent.name}, version: {agent.version}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deploy_agent()
