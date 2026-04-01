"""Knowledge graph API endpoints — stats, entities, graph visualization."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# vis-network group mapping
NODE_GROUP_MAP = {
    "Person": 1,
    "Organization": 2,
    "Event": 3,
    "Market": 4,
    "Policy": 5,
    "Location": 6,
    "LegalCase": 7,
}

# Edge colors by relationship type
EDGE_COLOR_MAP = {
    "AFFECTS": "#ef4444",
    "LEADS": "#3b82f6",
    "SPONSORS": "#22c55e",
    "ALLIED_WITH": "#f59e0b",
}
DEFAULT_EDGE_COLOR = "#6b7280"


@router.get("/stats")
async def graph_stats(request: Request) -> dict:
    """Get knowledge graph statistics."""
    stats = await request.app.state.neo4j.get_stats()
    return stats


@router.get("/entities/{entity_type}")
async def list_entities(request: Request, entity_type: str, limit: int = 50) -> dict:
    """List entities of a given type."""
    entities = await request.app.state.neo4j.get_entities(entity_type, limit)
    return {"entity_type": entity_type, "count": len(entities), "entities": entities}


@router.get("/graph-snapshot")
async def graph_snapshot(request: Request, limit: int = 200) -> dict:
    """Return vis-network compatible graph JSON — top nodes by connection count.

    Response format:
        { nodes: [{id, label, group, title, value}],
          edges: [{from, to, label, color}] }
    """
    neo4j = request.app.state.neo4j
    driver = neo4j.driver

    nodes = []
    edges = []
    seen_node_ids: set[int] = set()

    async with driver.session() as session:
        # Top N most-connected nodes
        result = await session.run(
            "MATCH (n)-[r]-() "
            "WITH n, labels(n)[0] AS label, count(r) AS degree "
            "ORDER BY degree DESC LIMIT $limit "
            "RETURN elementId(n) AS id, n, label, degree",
            {"limit": limit},
        )
        async for record in result:
            node_id = record["id"]
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            props = dict(record["n"])
            label_type = record["label"]
            name = props.get("name") or props.get("question") or props.get("case_name") or str(node_id)
            nodes.append({
                "id": node_id,
                "label": name[:40],
                "group": NODE_GROUP_MAP.get(label_type, 0),
                "title": f"{label_type}: {name}",
                "value": record["degree"],
            })

        # Edges between the selected nodes
        if seen_node_ids:
            result = await session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE elementId(a) IN $ids AND elementId(b) IN $ids "
                "RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel",
                {"ids": list(seen_node_ids)},
            )
            async for record in result:
                rel = record["rel"]
                edges.append({
                    "from": record["src"],
                    "to": record["tgt"],
                    "label": rel,
                    "color": EDGE_COLOR_MAP.get(rel, DEFAULT_EDGE_COLOR),
                })

    return {"nodes": nodes, "edges": edges}


@router.get("/search")
async def search_graph(request: Request, q: str, limit: int = 50) -> dict:
    """Search nodes by name and return matching subgraph."""
    neo4j = request.app.state.neo4j
    driver = neo4j.driver

    nodes = []
    edges = []
    seen_node_ids: set[int] = set()

    async with driver.session() as session:
        # Full-text search on entity names
        result = await session.run(
            "CALL db.index.fulltext.queryNodes('entity_names', $term) "
            "YIELD node, score "
            "WITH node, score ORDER BY score DESC LIMIT $limit "
            "OPTIONAL MATCH (node)-[r]-(neighbor) "
            "RETURN elementId(node) AS nid, node, labels(node)[0] AS nlabel, "
            "       elementId(neighbor) AS mid, neighbor, labels(neighbor)[0] AS mlabel, "
            "       type(r) AS rel, startNode(r) = node AS outgoing",
            {"term": q, "limit": limit},
        )
        async for record in result:
            # Source node
            nid = record["nid"]
            if nid not in seen_node_ids:
                seen_node_ids.add(nid)
                props = dict(record["node"])
                name = props.get("name") or props.get("question") or str(nid)
                nodes.append({
                    "id": nid,
                    "label": name[:40],
                    "group": NODE_GROUP_MAP.get(record["nlabel"], 0),
                    "title": f"{record['nlabel']}: {name}",
                    "value": 5,
                })

            # Neighbor node
            if record["mid"] is not None:
                mid = record["mid"]
                if mid not in seen_node_ids:
                    seen_node_ids.add(mid)
                    nprops = dict(record["neighbor"])
                    nname = nprops.get("name") or nprops.get("question") or str(mid)
                    nodes.append({
                        "id": mid,
                        "label": nname[:40],
                        "group": NODE_GROUP_MAP.get(record["mlabel"], 0),
                        "title": f"{record['mlabel']}: {nname}",
                        "value": 2,
                    })

                rel = record["rel"]
                src = nid if record["outgoing"] else mid
                tgt = mid if record["outgoing"] else nid
                edges.append({
                    "from": src,
                    "to": tgt,
                    "label": rel,
                    "color": EDGE_COLOR_MAP.get(rel, DEFAULT_EDGE_COLOR),
                })
