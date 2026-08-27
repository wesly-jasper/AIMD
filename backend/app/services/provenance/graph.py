from typing import List, Dict, Any, Optional

class ProvenanceNode:
    def __init__(self, node_id: str, label: str, metadata: Dict[str, Any] = None):
        self.id = node_id
        self.label = label
        self.metadata = metadata or {}
        
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "metadata": self.metadata
        }

class ProvenanceEdge:
    def __init__(self, source_id: str, target_id: str, relationship: str, confidence: float):
        self.source = source_id
        self.target = target_id
        self.relationship = relationship
        self.confidence = confidence
        
    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "confidence": self.confidence
        }

class ProvenanceGraph:
    def __init__(self):
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        
    def add_node(self, node: ProvenanceNode):
        self.nodes[node.id] = node
        
    def add_edge(self, edge: ProvenanceEdge):
        self.edges.append(edge)
        
    def to_dict(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }

def build_provenance_graph(uploaded_media_id: str, source_trace_results: Dict[str, Any]) -> ProvenanceGraph:
    """
    Constructs a provenance graph from source tracing results.
    """
    graph = ProvenanceGraph()
    
    # Add the uploaded media node
    graph.add_node(ProvenanceNode(
        node_id=uploaded_media_id, 
        label="Uploaded Media", 
        metadata={"status": "query"}
    ))
    
    earliest = source_trace_results.get("earliest_known_occurrence")
    if earliest:
        earliest_id = earliest["media_id"]
        graph.add_node(ProvenanceNode(
            node_id=earliest_id,
            label="Earliest Known Source",
            metadata={"timestamp": earliest["source_timestamp"]}
        ))
        
        # Link earliest to uploaded
        graph.add_edge(ProvenanceEdge(
            source_id=earliest_id,
            target_id=uploaded_media_id,
            relationship="derived_from",
            confidence=earliest["confidence"]
        ))
        
    # Could add other candidate nodes here as intermediate nodes if needed
    for candidate in source_trace_results.get("candidates", []):
        cand_id = candidate["media_id"]
        if cand_id != uploaded_media_id and (not earliest or cand_id != earliest["media_id"]):
            graph.add_node(ProvenanceNode(
                node_id=cand_id,
                label="Visually Similar Match",
                metadata={"timestamp": candidate["source_timestamp"]}
            ))
            
            graph.add_edge(ProvenanceEdge(
                source_id=cand_id,
                target_id=uploaded_media_id,
                relationship="visually_similar_to",
                confidence=candidate["confidence"]
            ))
            
    return graph
