# Gremlin Graph Visualizer — How It Works

## Architecture Overview

```
Browser (HTML/JS)
     ↕  HTTP (fetch)
Flask app (graph_viz/app.py)
     ↕  WebSocket (gremlinpython)
CosmosDB Gremlin API
```

Two processes communicate: the browser talks to Flask over HTTP, and Flask talks to CosmosDB over a persistent WebSocket.

---

## 1. The CosmosDB Connection (`app.py`)

```python
_state = {
    "endpoint": "cosmosdb-gremlin-abpatra.gremlin.cosmos.azure.com",
    "key":      "15F9Uz...",
    "database": "indigokg",
    "graph":    "knowledgegraph",
    "client":   None,
}
```

`_state` is a module-level dict that persists for the lifetime of the Flask process. It holds the Gremlin client, which is a **persistent WebSocket connection** — not a new connection per query.

```python
def _build_client(endpoint, key, database, graph):
    return gremlin_client.Client(
        url=f"wss://{endpoint}:443/",
        username=f"/dbs/{database}/colls/{graph}",
        password=key,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )
```

The `wss://` prefix means **WebSocket Secure**. CosmosDB speaks the Apache TinkerPop Gremlin Server protocol over this WebSocket. `GraphSONSerializersV2d0` is the wire format — it tells both sides how to serialize graph objects to/from JSON.

```python
def _run(query: str, timeout: int = 25) -> list:
    def _exec():
        return client.submitAsync(query).result().all().result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_exec).result(timeout=timeout)
```

`submitAsync` sends the Gremlin query string over the WebSocket and returns a **Future** immediately (non-blocking). `.result()` on that Future gives back a `ResultSet`. `.all().result()` blocks until all result pages arrive and returns a flat Python list.

The `ThreadPoolExecutor` wrapper exists purely to **enforce a timeout** — Python's gremlinpython library has no built-in query timeout, so we run it in a worker thread and use `future.result(timeout=25)` which raises `TimeoutError` if the thread doesn't finish in time.

---

## 2. The Result Parser (`parse_results`)

CosmosDB can return four fundamentally different Python types from a Gremlin query:

```python
def parse_results(results: list) -> dict:
    first = results[0]

    if isinstance(first, Vertex):       # proper gremlinpython objects (rare with CosmosDB)
    elif isinstance(first, Edge):
    elif isinstance(first, dict):
        if "inV" in first or "outV" in first:  # CosmosDB edge dict
        elif first.get("type") == "vertex":    # CosmosDB vertex dict
        else:                                   # .valueMap() result → table
    else:                                       # scalar (string/int) → table
```

### Why does CosmosDB return dicts instead of Vertex/Edge objects?

The `gremlinpython` library is designed for Apache TinkerPop servers. It deserializes GraphSON wire format into proper `Vertex`/`Edge` Python objects. But CosmosDB's Gremlin implementation has diverged from the standard — it returns vertices as plain JSON dicts:

```python
# CosmosDB returns this for g.V():
{
  "id": "domainentity__maintenance",
  "label": "DomainEntity",
  "type": "vertex",
  "properties": {
    "name": [{"id": "p0", "value": "maintenance"}],
    "category": [{"id": "p1", "value": "domain"}]
  }
}
```

Notice the `properties` format: each property is a **list of objects** with `id` and `value`. This is GraphSON's vertex-property format. The `_cosmos_props` function flattens this:

```python
def _cosmos_props(raw: dict) -> dict:
    for k, vlist in raw.items():
        if isinstance(vlist[0], dict):
            out[k] = vlist[0].get("value", "")  # unwrap {"id":..., "value": "maintenance"}
        else:
            out[k] = vlist[0]                    # unwrap ["maintenance"] (valueMap style)
```

For edges, CosmosDB returns:
```python
{
  "id": "e-123",
  "label": "RELATES_TO",
  "type": "edge",
  "outV": "domainentity__maintenance",   # source vertex ID
  "inV":  "domainentity__aircraft",      # target vertex ID
}
```

The parser detects edge dicts by checking `"inV" in first or "outV" in first`.

The parser outputs a normalised dict that the route handler uses:
```python
{"nodes": [...], "edges": [], "_auto_edges": True}
# or
{"nodes": [], "edges": [...], "_auto_vertices": ["id1", "id2", ...]}
# or
{"table": [...], "mode": "table"}
```

---

## 3. The Auto-Expand Logic (`/api/graph` route)

```python
graph = parse_results(results)

if graph.pop("_auto_edges", False) and graph.get("nodes"):
    ids = [n["id"] for n in graph["nodes"]]
    graph["edges"] = _fetch_edges_for_nodes(ids)

elif vids := graph.pop("_auto_vertices", None):
    graph["nodes"] = _fetch_vertices_for_ids(vids)
```

### Case 1 — User ran `g.V().hasLabel('DomainEntity')`

Parser returns nodes + `_auto_edges: True`. The route then fetches connected edges:

```python
def _fetch_edges_for_nodes(node_ids):
    edges = {}                          # dict, not list — for deduplication
    for i in range(0, len(node_ids), 20):       # batches of 20, ALL nodes
        batch = node_ids[i:i+20]
        ids_arg = ", ".join(f"'{_esc(v)}'" for v in batch)
        res = _run(f"g.V({ids_arg}).bothE().limit(500)")
        for e in parse_results(res).get("edges", []):
            if e["from"] in node_set and e["to"] in node_set:
                edges[e["id"]] = e      # keyed by ID → deduplicates
```

**Why batches of 20?** Gremlin query strings have practical length limits.

**Why a dict?** `bothE()` fetches edges from **both directions** of each node — if node A and node B are both in your result set and in different batches, the edge A→B gets returned twice (once from A's batch, once from B's). The dict keyed by edge ID silently deduplicates.

**Why no cap?** An earlier version capped at 60 nodes. This meant nodes beyond position 60 (e.g. Subject and Document nodes linked via MENTIONS edges) never had their edges fetched and appeared disconnected. The cap was removed so every node gets its edges regardless of how many nodes were returned.

### Case 2 — User ran `g.E().hasLabel('RELATES_TO')`

Parser returns edges + `_auto_vertices: [list of vertex IDs]`. The route fetches the endpoint vertices:

```python
def _fetch_vertices_for_ids(vids):
    nodes = []
    seen = set()
    for i in range(0, min(len(vids), 100), 20):
        ids_arg = ", ".join(f"'{_esc(v)}'" for v in batch)
        res = _run(f"g.V({ids_arg})")
        for n in parse_results(res).get("nodes", []):
            if n["id"] not in seen:
                seen.add(n["id"])
                nodes.append(n)
    # Stub any IDs CosmosDB didn't return
    for vid in vids:
        if vid not in seen:
            nodes.append(_node(vid, "unknown", {}))
```

The stubs exist because CosmosDB might not return a vertex (e.g. if it was deleted) — without stubs, edges would reference node IDs that don't exist in the vis.js DataSet, causing silent rendering failures.

---

## 4. The Frontend (`index.html`)

### Schema Loading

```javascript
async function loadSchema() {
    const data = await fetch("/api/schema").then(r => r.json());
    schemaData = data;
    renderSchema();    // populate left sidebar
    renderPresets();   // generate toolbar buttons
}
```

`/api/schema` runs `g.V().label().dedup()` and `g.E().label().dedup()` — two cheap Gremlin queries that return all distinct label strings. Everything downstream (buttons, colors, legend) is built from this dynamically. The app works with **any** Gremlin graph — no hardcoded labels.

### Color Assignment

```javascript
const PALETTE = [
  { background:"#1e2d5f", border:"#6c8cff", highlight:{...}, hover:{...} },
  { background:"#1a3d28", border:"#4ade80", ... },
  // 8 colors total
];
const colorMap = {};

function getColor(label) {
    if (!colorMap[label]) {
        colorMap[label] = PALETTE[Object.keys(colorMap).length % PALETTE.length];
        network.setOptions({ groups: buildGroups() });  // live-update vis.js
    }
    return colorMap[label];
}
```

The first time a new label is seen, it gets the next color from the palette (round-robin). `network.setOptions({ groups: ... })` pushes the update to vis.js immediately so newly colored nodes render correctly without a page reload.

### Vis.js DataSet and Network

```javascript
const nodesDS = new vis.DataSet([]);
const edgesDS = new vis.DataSet([]);

network = new vis.Network(
    document.getElementById("network"),
    { nodes: nodesDS, edges: edgesDS },
    VIS_OPTIONS
);
```

`vis.DataSet` is an **observable in-memory store**. The network subscribes to it — any `add`, `update`, or `remove` operation on the DataSet immediately re-renders the graph. This is why `nodesDS.clear()` then `nodesDS.add(...)` triggers a full graph redraw.

The `groups` config in `VIS_OPTIONS` maps label → color:
```javascript
groups: {
    "DomainEntity": { color: { background:"#1e2d5f", border:"#6c8cff", ... } },
    "Subject":      { color: { background:"#1a3d28", border:"#4ade80", ... } },
}
```
Every node has a `group` property set to its label string. Vis.js automatically applies the matching group's color. This is the entire color system — no CSS, no manual per-node styling.

### The Loading Fix

```javascript
// What didn't work:
network.on("stabilizationIterationsDone", hideLoading);
// This only fires during vis.js's initial startup stabilization phase.
// When you call nodesDS.add() on a running simulation, vis.js just
// continues the physics loop — no stabilization event fires again.

// What works:
hideLoading();  // called immediately after nodesDS.add()
network.fit();  // zoom to fit the graph viewport
```

The graph becomes interactive immediately while the physics simulation runs live in the background — the same behaviour as Neo4j Browser.

A 30-second safety-net timeout on `showLoading` ensures the overlay can never get permanently stuck regardless of what happens:
```javascript
_loadingTimer = setTimeout(hideLoading, 30000);
```

### The Tooltip Fix

```javascript
function makeTip(html) {
    const div = document.createElement("div");
    div.style.cssText = "background:#1e2130; ...";
    div.innerHTML = html;   // parses the HTML string into real DOM nodes
    return div;             // DOM element, not string
}

// Applied before adding nodes to the DataSet:
.map(n => ({ ...n, title: makeTip(n.title) }))
```

Vis.js 9.x changed tooltip behaviour: if `title` is a **string**, it is treated as plain text (HTML tags are escaped and shown literally). If `title` is a **DOM element**, vis.js inserts it directly into the tooltip container via `appendChild` — HTML renders correctly. Converting on the frontend means the backend keeps generating clean HTML; `makeTip` bridges the two.

---

## 5. Extended Result Type Coverage

### The Problem

Gremlin has many terminal steps beyond `g.V()` and `g.E()`. Each returns a different Python type, and the visualizer needs to handle all of them gracefully:

| Query pattern | Returns | Handling |
|---|---|---|
| `g.V()`, `g.V().hasLabel(...)` | `Vertex` objects | Graph — auto-fetch edges |
| `g.E()`, `g.E().hasLabel(...)` | `Edge` objects | Graph — auto-fetch vertices |
| `...path()` | `Path` objects | Graph — unpack each path |
| `...select('a','b','c')` | `dict` with Vertex/Edge **values** | Graph — extract from map values |
| `union(identity(), outE())` | **mixed** Vertex+Edge in same list | Graph — each item handled independently |
| `g.V().properties()` | `VertexProperty` objects | Table |
| `...valueMap()`, `...project()` | `dict` with scalar values | Table |
| `...count()`, `...values()`, `...label()` | `int` / `str` scalars | Table |

### The `_extract_element` Helper

Rather than writing a separate handler for every case, a single recursive function handles any Gremlin result element:

```python
def _extract_element(obj, nodes_map: dict, edge_map: dict) -> bool:
    if isinstance(obj, Vertex):
        # extract into nodes_map
    elif isinstance(obj, Edge):
        # extract into edge_map
    elif isinstance(obj, Path):
        for item in obj.objects:
            _extract_element(item, nodes_map, edge_map)   # recurse
    elif isinstance(obj, dict):
        if "labels" in obj and "objects" in obj:          # CosmosDB path dict
            for item in obj["objects"]:
                _extract_element(item, nodes_map, edge_map)
        elif "inV" in obj or "outV" in obj:               # CosmosDB edge dict
            ...
        elif obj.get("type") == "vertex" or ...:          # CosmosDB vertex dict
            ...
        else:                                             # .select() map
            for v in obj.values():
                _extract_element(v, nodes_map, edge_map)  # recurse into values
```

The key insight is the **last branch**: a `.select()` result is a dict whose *values* are vertices and edges. By recursing into `obj.values()`, the same function handles both direct graph elements and graph elements wrapped in named maps.

### How `parse_results` Uses It

```python
def _looks_like_graph(item) -> bool:
    """Quick check before committing to full extraction."""
    if isinstance(item, (Vertex, Edge, Path)):
        return True
    if isinstance(item, dict):
        return (
            "inV" in item or "outV" in item
            or item.get("type") in ("vertex", "edge")
            or ("labels" in item and "objects" in item)  # path dict
            or isinstance(item.get("properties"), dict)  # vertex dict
            or any(_looks_like_graph(v) for v in item.values())  # select map
        )
    return False
```

`parse_results` checks the first result item. If it looks like it contains graph elements, it runs `_extract_element` across all results, then decides what to auto-fetch:

```python
if _looks_like_graph(first):
    for item in results:
        _extract_element(item, nodes_map, edge_map)

    if nodes and not edges:
        return {..., "_auto_edges": True}      # g.V() pattern — fetch edges
    elif edges and not nodes:
        return {..., "_auto_vertices": vids}   # g.E() pattern — fetch vertices
    else:
        return {nodes, edges}                  # path/select — already complete
```

When **both** nodes and edges are present in the result (path queries, select queries), no auto-fetch is needed — the result already contains the full subgraph. When only one is present, the auto-expand logic fetches the other half.

### `VertexProperty` — Special Case

`g.V().properties()` or `g.V().properties('name')` returns `VertexProperty` objects — not vertices. These have `.key`, `.value`, and `.id` but no graph connectivity, so they render as a table:

```python
if isinstance(first, VertexProperty):
    table = [{"key": vp.key, "value": str(vp.value), "id": str(vp.id)}
             for vp in results]
    return {"table": table, "mode": "table", ...}
```

---

## 6. Sample Queries

> Paste any of these directly into the query bar. All path queries render as a full graph — nodes and edges included.

---

### Graph layers — quick reference

```
Document  ──MENTIONS──▶  Subject  ──CORRESPONDS_TO──▶  DomainEntity
                         Subject  ──RELATES_TO──────▶  Subject  (SPO triplets)
Document  ──HAS_SUBJECT──▶  Subject
```

`Subject` is the bridge layer. It connects unstructured document knowledge (lexical graph) to structured domain entities (domain graph).

---

### A. Full graph

Show every node and every edge in the graph:
```gremlin
g.V()
```

---

### B. Domain graph only

All structured domain entities and the relationships between them:
```gremlin
g.V().hasLabel('DomainEntity')
```

Just the RELATES_TO edges between domain entities:
```gremlin
g.E().hasLabel('RELATES_TO').limit(50)
```

---

### C. Lexical graph only

All subject nodes (auto-expands to show RELATES_TO + MENTIONS + HAS_SUBJECT edges):
```gremlin
g.V().hasLabel('Subject').limit(25)
```

All document nodes and their subject links:
```gremlin
g.V().hasLabel('Document')
```

Documents AND subjects together in one query:
```gremlin
g.V().hasLabel('Subject', 'Document').limit(40)
```

---

### D. How Subject bridges Document → DomainEntity

This is the key question: how does an unstructured document connect to a structured domain concept?

**Full bridge path** — walk from Document through Subject to DomainEntity and record every step:
```gremlin
g.V().hasLabel('Document').outE('MENTIONS').inV().outE('CORRESPONDS_TO').inV().path().limit(50)
```
Each path is: `Document → MENTIONS → Subject → CORRESPONDS_TO → DomainEntity`

**Same bridge in reverse** — start from DomainEntity and trace back to source documents:
```gremlin
g.V().hasLabel('DomainEntity').inE('CORRESPONDS_TO').outV().inE('MENTIONS').outV().path().limit(50)
```
Each path is: `DomainEntity → CORRESPONDS_TO → Subject → MENTIONS → Document`

**Named components with select** — label each part of the bridge explicitly:
```gremlin
g.V().hasLabel('Document').as('doc').out('MENTIONS').as('subject').out('CORRESPONDS_TO').as('domain').select('doc','subject','domain').limit(30)
```
Renders the same three-node chain but with named roles — useful when you want to inspect which specific document, subject, and domain entity are linked.

**Subject-centric view** — show all Subjects and auto-expand to see both what documents mention them AND which domain entities they map to:
```gremlin
g.V().hasLabel('Subject').limit(20)
```
Because auto-expand fetches `bothE()`, each Subject node will show MENTIONS edges (to Documents) and CORRESPONDS_TO edges (to DomainEntities) simultaneously — the clearest single-query view of the bridge.

---

### E. Targeted bridge queries (specific entities)

Find all documents that mention subjects corresponding to a specific domain entity:
```gremlin
g.V().hasLabel('DomainEntity').has('name', 'flights').in('CORRESPONDS_TO').in('MENTIONS').path().limit(30)
```

Find all domain entities reachable from a specific document:
```gremlin
g.V().hasLabel('Document').has('name', 'crew_operations').out('MENTIONS').out('CORRESPONDS_TO').path().limit(30)
```

Show everything connected to a specific subject by name:
```gremlin
g.V().hasLabel('Subject').has('name', 'crew').bothE().path().limit(30)
```

---

### F. Correspondence edges only

Just the CORRESPONDS_TO edges — each edge connects a Subject to a DomainEntity:
```gremlin
g.E().hasLabel('CORRESPONDS_TO').limit(50)
```

Just the MENTIONS edges — each edge connects a Document to a Subject:
```gremlin
g.E().hasLabel('MENTIONS').limit(50)
```

---

### G. Scalar / inspection queries (render as table)

These return property values, not graph elements — the visualizer shows them as a table automatically.

Count all vertices by label:
```gremlin
g.V().groupCount().by(label)
```

List all subject names:
```gremlin
g.V().hasLabel('Subject').values('name')
```

Get full properties of domain entities:
```gremlin
g.V().hasLabel('DomainEntity').valueMap('name', 'description', 'domain')
```

Count how many documents mention each subject:
```gremlin
g.V().hasLabel('Subject').project('subject','doc_count').by('name').by(__.in('MENTIONS').count()).limit(20)
```

---

### H. Query patterns and what they return

| Pattern | Returns | Renders as |
|---|---|---|
| `g.V()` | Vertex objects | Graph + auto-fetched edges |
| `g.E()` | Edge objects | Graph + auto-fetched vertices |
| `...path()` | Path objects (full traversal history) | Graph (nodes + edges extracted from path) |
| `...select('a','b','c')` | Maps with named vertex/edge values | Graph (extracted from map values) |
| `...valueMap()` | Dicts with property lists | Table |
| `...values('name')` | String list | Table |
| `...count()` | Integer | Table |
| `...groupCount()` | Dict of label → count | Table |
| `g.V().properties()` | VertexProperty objects | Table |

---

## 7. Data Flow — End to End

```
User clicks "DomainEntity" preset
  → setQuery("g.V().hasLabel('DomainEntity')")
  → POST /api/graph  { query: "g.V().hasLabel('DomainEntity')" }

Flask:
  1. _run(query)               → CosmosDB returns list of vertex dicts
  2. parse_results(...)        → { nodes: [...], _auto_edges: True }
  3. _fetch_edges_for_nodes()  → batched g.V(ids).bothE() calls → deduped edge dict
  4. return JSON               → { nodes: [...], edges: [...], stats: {...} }

Browser:
  1. Deduplicate nodes/edges by ID (CosmosDB can return the same element twice)
  2. Convert title strings → DOM elements via makeTip()
  3. getColor(label) for each node → assigns/retrieves palette color
  4. nodesDS.clear() + nodesDS.add(nodes)
  5. edgesDS.clear() + edgesDS.add(edges)
  6. hideLoading() → graph is now interactive
  7. network.fit() → zoom to show all nodes
  8. Physics simulation runs → nodes repel/attract into readable layout
  9. Click node → showProps(node.rawProps) → right panel populated
```

---

## 6. Key Design Decisions

| Decision | Why |
|---|---|
| Persistent WebSocket connection | Avoids reconnection overhead on every query |
| `ThreadPoolExecutor` for timeout | `gremlinpython` has no native query timeout |
| Dict (not list) for edge dedup | `bothE()` returns same edge from both endpoints when both are in the result set |
| No node cap in `_fetch_edges_for_nodes` | Earlier 60-node cap caused MENTIONS/HAS_SUBJECT edges to be silently dropped |
| `_extract_element` recursive helper | Single function handles Vertex, Edge, Path, select maps, mixed lists, and all CosmosDB dict variants — avoids a growing chain of `elif` branches for each new result type |
| `_looks_like_graph` pre-check | Avoids running full extraction on valueMap/scalar results; also handles nested select maps by recursing into dict values |
| nodes-only → `_auto_edges`, edges-only → `_auto_vertices`, both → return as-is | Path and select queries already carry both sides of the graph; only pure vertex/edge queries need the second fetch |
| `hideLoading()` immediately after `nodesDS.add()` | `stabilizationIterationsDone` only fires during initial load, not on data updates |
| `makeTip()` returns DOM element | vis.js 9.x escapes HTML strings in tooltips; DOM elements bypass this |
| Schema-driven presets and colors | Makes the app general-purpose — works with any Gremlin graph, no hardcoded labels |
