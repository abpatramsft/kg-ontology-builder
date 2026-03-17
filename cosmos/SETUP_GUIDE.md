# Knowledge Graph with Azure Cosmos DB (Gremlin API) — Setup Guide

This document walks through every step taken to set up a knowledge graph backed by Azure Cosmos DB's Gremlin (graph) API, from provisioning Azure resources to running the Python script.

---

## Table of Contents

1. [Why Cosmos DB + Gremlin?](#1-why-cosmos-db--gremlin)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Create the Cosmos DB Account](#3-step-1--create-the-cosmos-db-account)
4. [Step 2 — Create the Gremlin Database](#4-step-2--create-the-gremlin-database)
5. [Step 3 — Create the Graph Container](#5-step-3--create-the-graph-container)
6. [Step 4 — Install Python Dependencies](#6-step-4--install-python-dependencies)
7. [Step 5 — Configure Connection (.env)](#7-step-5--configure-connection-env)
8. [Step 6 — Run the Script](#8-step-6--run-the-script)
9. [Project Structure](#9-project-structure)
10. [Key Concepts](#10-key-concepts)

---

## 1. Why Cosmos DB + Gremlin?

| Concern | Why Cosmos DB Gremlin? |
|---|---|
| **Graph data** | Knowledge graphs are naturally represented as vertices (entities) and edges (relationships). Gremlin is a standard graph traversal language designed exactly for this. |
| **Managed service** | Cosmos DB is fully managed — no servers to patch, automatic backups, global distribution if needed. |
| **Scalability** | Cosmos DB scales throughput (RU/s) and storage independently, so your graph can grow without redesign. |
| **Multi-model** | The same Cosmos DB account concept supports SQL, MongoDB, Cassandra, Table, and Gremlin APIs, so you can pick the one that fits your data shape. |

---

## 2. Prerequisites

- **Azure CLI** (`az`) installed and logged in (`az login`).
- **Python 3.9+** installed.
- An **Azure subscription** with an existing resource group (we used `rg-abpatra-7946`).

---

## 3. Step 1 — Create the Cosmos DB Account

```bash
az cosmosdb create \
  --name cosmosdb-gremlin-abpatra \
  --resource-group rg-abpatra-7946 \
  --capabilities EnableGremlin \
  --locations regionName=westus2 failoverPriority=0 isZoneRedundant=false \
  --default-consistency-level Session \
  -o table
```

### What's happening

| Flag | Purpose |
|---|---|
| `--name` | Globally unique name for the Cosmos DB account. |
| `--capabilities EnableGremlin` | Tells Cosmos DB to expose the **Gremlin (graph) API** instead of the default SQL API. This is set at account creation time and cannot be changed later. |
| `--locations regionName=westus2 ...` | Places the data in **West US 2**. We used this region because East US had availability-zone capacity constraints at the time. `isZoneRedundant=false` avoids the zone-redundancy quota issue. |
| `--default-consistency-level Session` | **Session consistency** means your own reads always see your own writes, while still being fast. This is the recommended default for most applications. |

> **Note:** Account creation takes 3–5 minutes. This is the longest step.

---

## 4. Step 2 — Create the Gremlin Database

```bash
az cosmosdb gremlin database create \
  --account-name cosmosdb-gremlin-abpatra \
  --resource-group rg-abpatra-7946 \
  --name knowledgeGraphDB \
  -o table
```

### What's happening

A **database** in Cosmos DB is a logical namespace that groups one or more containers (graphs). Think of it like a schema in a relational database. You can have multiple graphs inside one database.

---

## 5. Step 3 — Create the Graph Container

```bash
MSYS_NO_PATHCONV=1 az cosmosdb gremlin graph create \
  --account-name cosmosdb-gremlin-abpatra \
  --resource-group rg-abpatra-7946 \
  --database-name knowledgeGraphDB \
  --name knowledgeGraph \
  --partition-key-path "/category" \
  --throughput 400 \
  -o table
```

### What's happening

| Flag | Purpose |
|---|---|
| `--name knowledgeGraph` | The graph container name — this is where your vertices and edges live. |
| `--partition-key-path "/category"` | **Partition key** determines how Cosmos DB distributes data across physical partitions. We chose `/category` so that related entities (e.g., all "programming" items) are co-located, making queries within a category fast. |
| `--throughput 400` | Provisions **400 RU/s** (Request Units per second) — the minimum and cheapest option, perfect for learning and development. You can scale up later. |
| `MSYS_NO_PATHCONV=1` | A Git Bash workaround — without this, Git Bash on Windows converts `/category` into a Windows file path like `C:/Program Files/Git/category`, which Cosmos DB rejects. |

### About Partition Keys

- Every vertex you insert must include the partition key property (`category` in our case).
- Cosmos DB uses it to decide which physical partition stores the data.
- Good partition keys have **high cardinality** (many distinct values) and **align with your query patterns**.

---

## 6. Step 4 — Install Python Dependencies

```bash
pip install gremlinpython python-dotenv
```

| Package | Purpose |
|---|---|
| `gremlinpython` | Official Apache TinkerPop Gremlin client for Python. Communicates with the Cosmos DB Gremlin endpoint over WebSockets. |
| `python-dotenv` | Loads environment variables from a `.env` file so we don't hardcode secrets in code. |

---

## 7. Step 5 — Configure Connection (.env)

Create a `.env` file in the project root:

```
COSMOS_DB_ENDPOINT=cosmosdb-gremlin-abpatra.gremlin.cosmos.azure.com
COSMOS_DB_KEY=<your-primary-key>
COSMOS_DB_DATABASE=knowledgeGraphDB
COSMOS_DB_GRAPH=knowledgeGraph
```

### Getting the primary key

```bash
az cosmosdb keys list \
  --name cosmosdb-gremlin-abpatra \
  --resource-group rg-abpatra-7946 \
  --query "primaryKey" -o tsv
```

Or from the **Azure Portal**: navigate to your Cosmos DB account → **Settings** → **Keys** → copy **PRIMARY KEY**.

> **Security:** The `.env` file is listed in `.gitignore` to prevent accidental commits of secrets.

---

## 8. Step 6 — Run the Script

```bash
python db_script.py
```

### What the script does

1. **Connects** to the Cosmos DB Gremlin endpoint using WebSockets (`wss://`).
2. **Cleans up** any existing data (`g.V().drop()`).
3. **Adds 7 vertices** — entities like programming languages, frameworks, databases, and CS concepts.
4. **Adds 6 edges** — relationships like `has_framework`, `can_use`, `implements`.
5. **Runs sample queries** to traverse the graph:
   - List all vertices
   - List all edges
   - Find frameworks for Python
   - Find databases that implement Graph Theory
   - Trace a full path: Python → framework → database

### The knowledge graph looks like this

```
Python ──has_framework──▶ Django ──can_use──▶ Cosmos DB ──implements──▶ Graph Theory
                                                                           ▲
JavaScript ──has_framework──▶ React ──can_use──┘        Neo4j ──implements──┘
```

---

## 9. Project Structure

```
cosmos/
├── .env              # Connection secrets (git-ignored)
├── .gitignore        # Keeps .env out of version control
├── db_script.py      # Main script — builds and queries the knowledge graph
└── SETUP_GUIDE.md    # This file
```

---

## 10. Key Concepts

### Gremlin Query Language — Quick Reference

| Query | Meaning |
|---|---|
| `g.addV('label').property('id','x').property('key','val')` | Add a vertex with a label and properties |
| `g.V('id').addE('relationship').to(g.V('other_id'))` | Add an edge between two vertices |
| `g.V()` | Get all vertices |
| `g.E()` | Get all edges |
| `g.V('id').out('edge_label')` | Traverse outgoing edges from a vertex |
| `g.V('id').in('edge_label')` | Traverse incoming edges to a vertex |
| `g.V('id').out().out()` | Multi-hop traversal |
| `g.V().valueMap(true)` | Get all properties of all vertices |
| `g.V().drop()` | Delete all vertices (and their edges) |

### Cosmos DB Terminology

| Term | Meaning |
|---|---|
| **Account** | Top-level resource; determines the API type (Gremlin, SQL, etc.) |
| **Database** | Logical grouping of containers/graphs |
| **Graph (Container)** | Stores vertices and edges; has a partition key and provisioned throughput |
| **RU/s** | Request Units per second — Cosmos DB's currency for throughput. A simple point-read costs ~1 RU. |
| **Partition Key** | Property used to distribute data across physical partitions |
| **Consistency Level** | Trade-off between read freshness and performance (Session is the recommended default) |

---

## Next Steps

- **Azure Portal → Data Explorer**: You can visually browse and query your graph in the portal.
- **Add more entities**: Extend the `VERTICES` and `EDGES` lists in `db_script.py`.
- **Parameterized queries**: Use Gremlin bindings for user-provided values to avoid injection.
- **Autoscale throughput**: Switch from fixed 400 RU/s to autoscale for production workloads.
