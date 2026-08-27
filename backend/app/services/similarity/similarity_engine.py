from typing import List, Dict, Any
from app.services.similarity.internet_search_provider import BaseInternetSearchProvider

class SimilarityEngine:
    def __init__(self, search_provider: BaseInternetSearchProvider = None):
        self.search_provider = search_provider

    def search(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Coordinates similarity search. Currently configured to search over the internet.
        """
        if not self.search_provider:
            raise RuntimeError("SimilarityEngine requires an InternetSearchProvider")
            
        # Here we would typically generate a fingerprint/embedding first, 
        # but reverse image search APIs often take the file directly.
        candidates = self.search_provider.search_by_image(file_path)
        
        # Sort by similarity
        candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return candidates
