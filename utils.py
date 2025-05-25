# utils.py

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Ensure the Qdrant collection exists
def ensure_collection_exists(qdrant_client: QdrantClient):
    try:
        qdrant_client.get_collection("Document")
    except Exception:
        qdrant_client.create_collection(
            collection_name="Document",
            vectors_config=VectorParams(size=1536, 
                                        distance="Cosine"),
        )
