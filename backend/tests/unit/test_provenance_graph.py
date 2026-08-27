from app.services.provenance.graph import (
    ProvenanceNode,
    ProvenanceEdge,
    ProvenanceGraph,
    build_provenance_graph
)

def test_provenance_graph_models():
    graph = ProvenanceGraph()
    node1 = ProvenanceNode("node1", "Original Image")
    node2 = ProvenanceNode("node2", "Manipulated Image")
    edge = ProvenanceEdge("node1", "node2", "derived_from", 0.95)
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge(edge)
    
    data = graph.to_dict()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["edges"][0]["relationship"] == "derived_from"

def test_build_provenance_graph():
    trace_results = {
        "earliest_known_occurrence": {
            "media_id": "source_123",
            "source_timestamp": "2024-01-01T00:00:00Z",
            "confidence": 0.9
        },
        "candidates": [
            {
                "media_id": "candidate_456",
                "source_timestamp": "2024-02-01T00:00:00Z",
                "confidence": 0.8
            }
        ]
    }
    
    graph = build_provenance_graph("uploaded_789", trace_results)
    data = graph.to_dict()
    
    node_ids = [n["id"] for n in data["nodes"]]
    assert "uploaded_789" in node_ids
    assert "source_123" in node_ids
    assert "candidate_456" in node_ids
    assert len(data["edges"]) == 2
