from load_data import load_documents
from yandex_embed import get_embedding
from pinecone_setup import index

BATCH_SIZE = 100


def main():
    docs = load_documents()
    vectors = []
    total_uploaded = 0
    skipped_files = []
    processed_files = 0
    failed_chunks = 0

    for doc_idx, doc in enumerate(docs, 1):
        metadata = doc.get("metadata", {}) or {}
        file_name = metadata.get("file_name", f"doc_{doc_idx}")
        folder = metadata.get("folder", "root")
        chunks = doc.get("chunks", []) or []

        if not chunks:
            skipped_files.append(file_name)
            continue

        processed_files += 1

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                failed_chunks += 1
                continue

            try:
                emb = get_embedding(chunk, kind="doc")
            except Exception as e:
                print(f"❌ Embedding failed for chunk {i} of {file_name}: {e}")
                failed_chunks += 1
                continue

            vector_id = f"{folder}_{file_name}_{i}"

            vectors.append(
                {
                    "id": vector_id,
                    "values": emb,
                    "metadata": {**metadata, "chunk_id": i, "text": chunk},
                }
            )

            if len(vectors) >= BATCH_SIZE:
                index.upsert(vectors=vectors)
                total_uploaded += len(vectors)
                vectors = []

    if vectors:
        index.upsert(vectors=vectors)
        total_uploaded += len(vectors)

    stats = index.describe_index_stats()
    print(f"✅ Pinecone stats: {stats}")
    print(
        f"📄 Documents processed: {processed_files}, "
        f"skipped: {len(skipped_files)}, "
        f"failed chunks: {failed_chunks}, "
        f"vectors uploaded: {total_uploaded}"
    )


if __name__ == "__main__":
    main()