"""
cosmos_setup_arm.py — Create Cosmos DB database & container via ARM REST API.

This bypasses the Cosmos DB data-plane SDK entirely and uses the Azure Resource
Manager (ARM) control plane instead. This is useful when:
  - The data-plane RBAC hasn't propagated yet
  - The az CLI version doesn't support --vector-policy
  - PowerShell JSON quoting is causing headaches

It uses your az login credentials (DefaultAzureCredential) to call the ARM API
directly, which only requires standard Azure RBAC (Contributor/Owner) on the
resource group — not Cosmos DB data-plane roles.

USAGE:
  python cosmos_setup_arm.py
"""

import requests
from azure.identity import DefaultAzureCredential

# ── Configuration ───────────────────────────────────────────────────────────
SUBSCRIPTION_ID = "ea5dfd63-50a1-4e2f-8144-5f6c7088c0c7"
RESOURCE_GROUP = "rg-abpatra-7946"
ACCOUNT_NAME = "cosmosdb-vectors"
DATABASE_NAME = "vector_demo_arm"
CONTAINER_NAME = "documents"

ARM_BASE = (
    f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.DocumentDB/databaseAccounts/{ACCOUNT_NAME}"
)
API_VERSION = "2024-05-15"  # ARM API version that supports vector policies


def get_arm_token() -> str:
    """Get a bearer token for ARM API calls."""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


def create_database(token: str):
    """Create the SQL database via ARM REST API."""
    url = f"{ARM_BASE}/sqlDatabases/{DATABASE_NAME}?api-version={API_VERSION}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "location": "East US",  # Must match your Cosmos account region
        "properties": {
            "resource": {
                "id": DATABASE_NAME
            }
        }
    }

    print(f"Creating database '{DATABASE_NAME}'...")
    resp = requests.put(url, json=body, headers=headers)

    if resp.status_code in (200, 201, 202):
        print(f"✓ Database '{DATABASE_NAME}' created/exists")
    else:
        print(f"✗ Failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()


def create_container(token: str):
    """Create the container with vector embedding policy and vector index via ARM."""
    url = (
        f"{ARM_BASE}/sqlDatabases/{DATABASE_NAME}"
        f"/containers/{CONTAINER_NAME}?api-version={API_VERSION}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "location": "East US",
        "properties": {
            "resource": {
                "id": CONTAINER_NAME,
                "partitionKey": {
                    "paths": ["/source"],
                    "kind": "Hash",
                    "version": 2,
                },
                "indexingPolicy": {
                    "indexingMode": "consistent",
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [
                        {"path": "/embedding/*"},
                        {"path": "/\"_etag\"/?"},
                    ],
                    "vectorIndexes": [
                        {"path": "/embedding", "type": "quantizedFlat"}
                    ],
                },
                "vectorEmbeddingPolicy": {
                    "vectorEmbeddings": [
                        {
                            "path": "/embedding",
                            "dataType": "float32",
                            "distanceFunction": "cosine",
                            "dimensions": 1536,
                        }
                    ]
                },
            }
        },
    }

    print(f"Creating container '{CONTAINER_NAME}' with vector policy...")
    resp = requests.put(url, json=body, headers=headers)

    if resp.status_code in (200, 201, 202):
        print(f"✓ Container '{CONTAINER_NAME}' created/exists")
        print(f"  Partition key : /source")
        print(f"  Vector index  : quantizedFlat on /embedding (1536 dims, cosine)")
    else:
        print(f"✗ Failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()


if __name__ == "__main__":
    token = get_arm_token()
    create_database(token)
    create_container(token)
    print("\n✅ Done! Database and container are ready.")
