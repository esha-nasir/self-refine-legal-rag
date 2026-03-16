import os

try:
    # Standard import path (works for most pinecone SDK installs)
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except ImportError:
    # Fallback for namespace-style installs on newer Python builds
    from pinecone.pinecone import Pinecone  # type: ignore
    from pinecone.models import ServerlessSpec  # type: ignore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing env var: PINECONE_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "judgements")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "256"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

listed = pc.list_indexes()
if hasattr(listed, "indexes"):
    existing = [i.name for i in listed.indexes]
else:
    existing = [i["name"] for i in listed]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

index = pc.Index(INDEX_NAME)
