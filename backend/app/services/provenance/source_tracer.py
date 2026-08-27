from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class SourceCandidate:
    def __init__(self, media_id: str, similarity: float, source_timestamp: Optional[str], discovery_timestamp: str, matching_method: str, confidence: float):
        self.media_id = media_id
        self.similarity = similarity
        self.source_timestamp = source_timestamp
        self.discovery_timestamp = discovery_timestamp
        self.matching_method = matching_method
        self.confidence = confidence
        
    def to_dict(self):
        return {
            "media_id": self.media_id,
            "similarity": self.similarity,
            "source_timestamp": self.source_timestamp,
            "discovery_timestamp": self.discovery_timestamp,
            "matching_method": self.matching_method,
            "confidence": self.confidence
        }

class SourceTracer:
    def __init__(self, similarity_engine):
        self.similarity_engine = similarity_engine

    def trace_source(self, file_path: str) -> Dict[str, Any]:
        """
        Traces the source of a media file by querying the similarity engine
        and ranking results to find the earliest known occurrence.
        """
        candidates_raw = self.similarity_engine.search(file_path)
        candidates = []
        
        for idx, c in enumerate(candidates_raw):
            candidates.append(SourceCandidate(
                media_id=c.get("url", f"unknown_source_{idx}"),
                similarity=c.get("similarity", 0.0),
                source_timestamp=c.get("timestamp"),
                discovery_timestamp=datetime.now(timezone.utc).isoformat(),
                matching_method=c.get("method", "unknown"),
                confidence=c.get("similarity", 0.0) # Using similarity as confidence baseline
            ))
            
        # Sort by timestamp to find the earliest occurrence
        # If timestamp is missing, it goes to the end
        def sort_key(c: SourceCandidate):
            if c.source_timestamp:
                try:
                    # Basic ISO parsing
                    return datetime.fromisoformat(c.source_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    pass
            return datetime.max
            
        candidates.sort(key=sort_key)
        
        earliest_source = candidates[0] if candidates else None
        
        return {
            "earliest_known_occurrence": earliest_source.to_dict() if earliest_source else None,
            "candidates": [c.to_dict() for c in candidates]
        }
