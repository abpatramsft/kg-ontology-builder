"""
graph_viz/app.py — Standalone general-purpose Gremlin graph visualizer.

Connects to any Gremlin-compatible graph database (CosmosDB, TinkerGraph, etc.)
and renders the graph interactively in the browser.

Usage:
    python graph_viz/app.py          # http://localhost:5051

Environment variables (loaded from ../.env or set manually in the UI):
    COSMOS_DB_ENDPOINT   Gremlin hostname (no wss:// prefix)
    COSMOS_DB_KEY        Auth key / password
    COSMOS_DB_DATABASE   Database name         (default: indigokg)
    COSMOS_DB_GRAPH      Graph / container     (default: knowledgegraph)
"""

import concurrent.futures
import json
import os
import threading

from flask import Flask, jsonify, render_template, request
from gremlin_python.driver import client as gremlin_client, serializer
from gremlin_python.structure.graph import Edge, Path, Vertex, VertexProperty

# ── bootstrap ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))


# ── connection state ───────────────────────────────────────────────────────
_lock = threading.Lock()
_state: dict = {
    "endpoint": os.getenv("COSMOS_DB_ENDPOINT", ""),
    "key":      os.getenv("COSMOS_DB_KEY", ""),
    "database": os.getenv("COSMOS_DB_DATABASE", "indigokg"),
    "graph":    os.getenv("COSMOS_DB_GRAPH", "knowledgegraph"),
    "client":   None,
    "connected": False,
}


def _build_client(endpoint: str, key: str, database: str, graph: str):
    return gremlin_client.Client(
        url=f"wss://{endpoint}:443/",
        traversal_source="g",
        username=f"/dbs/{database}/colls/{graph}",
        password=key,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def _run(query: str, timeout: int = 25) -> list:
    """Execute a Gremlin query and return the result list."""
    with _lock:
        client = _state.get("client")
    if client is None:
        raise RuntimeError("Not connected to any graph database.")

    def _exec():
        return client.submitAsync(query).result().all().result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_exec).result(timeout=timeout)


# ── helpers ────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape a value for embedding in a Gremlin single-quoted string."""
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def _cosmos_props(raw: dict) -> dict:
    """
    Flatten a CosmosDB vertex property dict.
    Handles both GraphSON vertex-property format:
      {"name": [{"id": "p0", "value": "flights"}]}
    and valueMap-style:
      {"name": ["flights"]}
    """
    out: dict = {}
    for k, vlist in raw.items():
        if not isinstance(vlist, list) or not vlist:
            out[k] = vlist
        elif isinstance(vlist[0], dict):
            out[k] = vlist[0].get("value", "")
        else:
            out[k] = vlist[0]
    return out


def _display_name(vid: str, props: dict) -> str:
    return (
        props.get("name")
        or props.get("title")
        or props.get("topic_summary", "")[:40]
        or vid[:35]
    )


def _node(vid: str, label: str, props: dict) -> dict:
    name = _display_name(vid, props)
    tip = "<br>".join(
        [f"<b>{label}</b>", f"<i>{vid}</i>"]
        + [f"{k}: {str(v)[:150]}" for k, v in list(props.items())[:20]]
    )
    return {
        "id":       vid,
        "label":    name,
        "title":    tip,
        "group":    label,
        "rawProps": {"_id": vid, "_label": label, **props},
    }


def _edge_dict(eid: str, out_id: str, in_id: str, label: str, props: dict) -> dict:
    return {
        "id":     eid,
        "from":   out_id,
        "to":     in_id,
        "label":  label,
        "arrows": "to",
        "title":  label + (f"<br>{json.dumps(props, default=str)[:200]}" if props else ""),
    }


# ── Gremlin result parser ──────────────────────────────────────────────────

def _extract_element(obj, nodes_map: dict, edge_map: dict) -> bool:
    """
    Recursively pull a vertex or edge out of any Gremlin result element.

    Handles gremlinpython objects (Vertex, Edge, Path), CosmosDB dicts
    (vertex dict, edge dict, path dict), and .select() maps whose values
    are themselves graph elements.  Returns True if anything was extracted.
    """
    if isinstance(obj, Vertex):
        vid = str(obj.id)
        if vid not in nodes_map:
            props = {vp.key: vp.value for vp in (obj.properties or [])}
            nodes_map[vid] = _node(vid, obj.label, props)
        return True

    elif isinstance(obj, Edge):
        eid = str(obj.id)
        if eid not in edge_map:
            props = {p.key: p.value for p in (obj.properties or [])}
            edge_map[eid] = _edge_dict(
                eid, str(obj.outV.id), str(obj.inV.id), obj.label, props
            )
        return True

    elif isinstance(obj, Path):
        # Use a list comprehension — NOT a generator — so every element is
        # processed before any() checks the results. A generator would
        # short-circuit on the first True and skip the rest of the path.
        results = [_extract_element(item, nodes_map, edge_map) for item in (obj.objects or [])]
        return any(results)

    elif isinstance(obj, dict):
        # CosmosDB path dict  {"labels": [...], "objects": [...]}
        if "labels" in obj and isinstance(obj.get("objects"), list):
            # Same reason: list comprehension, not generator
            results = [_extract_element(item, nodes_map, edge_map) for item in obj["objects"]]
            return any(results)

        # CosmosDB edge dict
        if "inV" in obj or "outV" in obj:
            eid = str(obj.get("id", ""))
            if eid not in edge_map:
                raw = obj.get("properties", {})
                props = _cosmos_props(raw) if isinstance(raw, dict) else {}
                edge_map[eid] = _edge_dict(
                    eid, str(obj.get("outV", "")), str(obj.get("inV", "")),
                    obj.get("label", ""), props
                )
            return True

        # CosmosDB vertex dict
        if obj.get("type") == "vertex" or (
            "id" in obj and "label" in obj and isinstance(obj.get("properties"), dict)
        ):
            vid = str(obj.get("id", ""))
            if vid not in nodes_map:
                raw = obj.get("properties", {})
                props = _cosmos_props(raw) if isinstance(raw, dict) else {}
                nodes_map[vid] = _node(vid, obj.get("label", "unknown"), props)
            return True

        # .select() map — dict whose values are themselves graph elements
        # e.g. {"src": vertex_dict, "e": edge_dict, "dst": vertex_dict}
        extracted = [_extract_element(v, nodes_map, edge_map) for v in obj.values()]
        return any(extracted)

    return False


def parse_results(results: list) -> dict:
    """
    Convert any Gremlin result list into a normalised dict:
      mode="graph"  → {nodes, edges, _auto_edges | _auto_vertices}
      mode="table"  → {table, warning}

    Handled result types
    ────────────────────
    Vertex / Edge              g.V() / g.E()
    Path                       ...path()
    VertexProperty             g.V().properties()
    select() maps              ...select('a','b','c') where values are graph elements
    Mixed Vertex+Edge list     g.V().union(identity(), outE())
    CosmosDB vertex/edge dicts same queries via CosmosDB's non-standard serialisation
    valueMap / project dicts   ...valueMap() / .project()  → table
    Scalars                    .count() / .values() / .label()  → table
    """
    if not results:
        return {"nodes": [], "edges": [], "mode": "graph",
                "warning": "Query returned no results."}

    first = results[0]
    nodes_map: dict = {}
    edge_map:  dict = {}

    # ── VertexProperty  (g.V().properties()) ────────────────────────────────
    if isinstance(first, VertexProperty):
        table = [{"key": vp.key, "value": str(vp.value), "id": str(vp.id)}
                 for vp in results]
        return {"nodes": [], "edges": [], "table": table, "mode": "table",
                "warning": "Showing vertex properties as table (g.V().properties())."}

    # ── Types that contain graph elements: Vertex, Edge, Path, select maps,
    #    mixed lists, CosmosDB dicts — all handled by _extract_element ───────
    graph_types = (Vertex, Edge, Path)

    def _looks_like_graph(item) -> bool:
        """Quick check: would _extract_element do anything useful with this?"""
        if isinstance(item, graph_types):
            return True
        if isinstance(item, dict):
            return (
                "inV" in item or "outV" in item
                or item.get("type") in ("vertex", "edge")
                or ("labels" in item and "objects" in item)
                or isinstance(item.get("properties"), dict)
                or any(_looks_like_graph(v) for v in item.values())
            )
        return False

    if _looks_like_graph(first):
        for item in results:
            _extract_element(item, nodes_map, edge_map)

        nodes = list(nodes_map.values())
        edges = list(edge_map.values())

        # If only nodes came back (no edges embedded in result), auto-fetch edges.
        # If only edges came back (no nodes), auto-fetch endpoint vertices.
        # If both came back (e.g. path / select), return as-is.
        if nodes and not edges:
            return {"nodes": nodes, "edges": [], "mode": "graph", "_auto_edges": True}
        elif edges and not nodes:
            vids = list({e["from"] for e in edges} | {e["to"] for e in edges})
            return {"nodes": [], "edges": edges, "mode": "graph",
                    "_auto_vertices": vids}
        else:
            return {"nodes": nodes, "edges": edges, "mode": "graph"}

    # ── Scalar results (.count(), .values(), .label(), …) ───────────────────
    if not isinstance(first, dict):
        return {
            "nodes": [], "edges": [],
            "table": [{"value": str(r)} for r in results],
            "mode": "table",
            "warning": (
                f"Query returned {type(first).__name__} scalars — displaying as table. "
                "Use g.V() or g.E() to visualize as a graph."
            ),
        }

    # ── Dict results that contain no graph elements (.valueMap(), .project()) ─
    sample_keys = list(first.keys())[:6]
    return {
        "nodes": [], "edges": [], "table": results, "mode": "table",
        "warning": (
            f"Query returned property maps (keys: {sample_keys}) — displaying as table. "
            "Remove .valueMap() / .project() to visualize as graph."
        ),
    }


def _fetch_edges_for_nodes(node_ids: list) -> list:
    """Fetch edges incident to a list of vertex IDs (batched, deduplicated).

    Processes ALL node IDs with no cap — every node gets its edges fetched
    so nothing appears disconnected (e.g. MENTIONS edges in the lexical graph).
    Uses a dict keyed by edge ID to deduplicate: bothE() returns the same edge
    from both endpoints when both are in the result set.
    """
    edges: dict = {}
    node_set = set(node_ids)
    for i in range(0, len(node_ids), 20):
        batch = node_ids[i : i + 20]
        ids_arg = ", ".join(f"'{_esc(v)}'" for v in batch)
        try:
            res = _run(f"g.V({ids_arg}).bothE().limit(500)", timeout=20)
            parsed = parse_results(res)
            for e in parsed.get("edges", []):
                if e["from"] in node_set and e["to"] in node_set:
                    edges[e["id"]] = e  # dedup by edge ID
        except Exception:
            pass
    return list(edges.values())


def _fetch_vertices_for_ids(vids: list) -> list:
    """Fetch vertex details for a list of IDs, with stubs for missing ones."""
    nodes: list = []
    seen: set = set()
    cap = min(len(vids), 100)
    for i in range(0, cap, 20):
        batch = vids[i : i + 20]
        ids_arg = ", ".join(f"'{_esc(v)}'" for v in batch)
        try:
            res = _run(f"g.V({ids_arg})", timeout=20)
            parsed = parse_results(res)
            parsed.pop("_auto_edges", None)
            for n in parsed.get("nodes", []):
                if n["id"] not in seen:
                    seen.add(n["id"])
                    nodes.append(n)
        except Exception:
            pass
    for vid in vids[:cap]:
        if vid not in seen:
            nodes.append(_node(vid, "unknown", {}))
            seen.add(vid)
    return nodes


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with _lock:
        s = dict(_state)
    return render_template("index.html",
                           endpoint=s["endpoint"],
                           database=s["database"],
                           graph_name=s["graph"],
                           connected=s["connected"])


@app.route("/api/connect", methods=["POST"])
def api_connect():
    body = request.get_json(force=True) or {}
    ep  = body.get("endpoint", "").strip()
    key = body.get("key", "").strip()
    db  = body.get("database", "").strip()
    gr  = body.get("graph", "").strip()

    if not all([ep, key, db, gr]):
        return jsonify({"error": "All connection fields are required."}), 400

    try:
        c = _build_client(ep, key, db, gr)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(
                lambda: c.submitAsync("g.V().limit(1).count()").result().all().result()
            ).result(timeout=15)
    except Exception as exc:
        return jsonify({"error": f"Connection failed: {exc}"}), 500

    with _lock:
        old = _state.get("client")
        if old:
            try: old.close()
            except Exception: pass
        _state.update({"endpoint": ep, "key": key, "database": db,
                       "graph": gr, "client": c, "connected": True})

    return jsonify({"ok": True, "message": f"Connected to {db}/{gr}"})


@app.route("/api/schema")
def api_schema():
    """Return vertex and edge labels for this graph (no counts — fast)."""
    try:
        vlabels = _run("g.V().label().dedup()", timeout=15)
        elabels = _run("g.E().label().dedup()", timeout=15)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"vertex_labels": sorted(vlabels),
                    "edge_labels":   sorted(elabels)})


@app.route("/api/graph", methods=["POST"])
def api_graph():
    """Execute a Gremlin query and return vis.js-compatible JSON."""
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    auto_expand = body.get("auto_expand", True)

    if not query:
        return jsonify({"error": "No query provided."}), 400

    write_kw = ("addv(", "adde(", ".drop()", ".drop(", ".property(")
    if any(kw in query.lower() for kw in write_kw):
        return jsonify({"error": "Write operations are not permitted."}), 400

    try:
        results = _run(query)
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "Query timed out (>25 s). Add a .limit() to reduce results."}), 504
    except Exception as exc:
        return jsonify({"error": f"Query failed: {exc}"}), 500

    graph = parse_results(results)

    if auto_expand:
        if graph.pop("_auto_edges", False) and graph.get("nodes"):
            ids = [n["id"] for n in graph["nodes"]]
            graph["edges"] = _fetch_edges_for_nodes(ids)

        elif vids := graph.pop("_auto_vertices", None):
            graph["nodes"] = _fetch_vertices_for_ids(vids)

    graph.pop("_auto_edges", None)
    graph.pop("_auto_vertices", None)

    return jsonify({
        "nodes":   graph.get("nodes", []),
        "edges":   graph.get("edges", []),
        "table":   graph.get("table"),
        "mode":    graph.get("mode", "graph"),
        "error":   graph.get("error"),
        "warning": graph.get("warning"),
        "stats": {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
        },
    })


# ── startup ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with _lock:
        ep  = _state["endpoint"]
        key = _state["key"]
        db  = _state["database"]
        gr  = _state["graph"]

    if ep and key:
        print("Auto-connecting from environment variables...")
        try:
            c = _build_client(ep, key, db, gr)
            with _lock:
                _state["client"] = c
                _state["connected"] = True
            print(f"  OK — {db}/{gr}")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("No credentials in environment — configure via the UI.")

    print("\n  Gremlin Graph Visualizer → http://localhost:5051\n")
    app.run(host="0.0.0.0", port=5051, debug=False, threaded=True)
