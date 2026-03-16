# self-refine-legal-rag

`self-refine-legal-rag` is an enhanced Retrieval-Augmented Generation (RAG) pipeline for legal question answering over court judgment PDFs.

The project extends a baseline legal RAG system with a self-refinement and critic-guided reasoning layer. It retrieves relevant legal evidence, evaluates candidate answers for relevance, grounding, and utility, and iteratively improves responses to reduce hallucinations and improve answer quality.

## Features

- Legal RAG pipeline for court judgment PDFs
- Native PDF text extraction with local OCR fallback
- Metadata-aware vector retrieval with Pinecone
- Adaptive retrieval inspired by Self-RAG
- Critic-guided evidence filtering
- Grounding and utility checks for generated answers
- Retry and post-check refinement steps
- Optional beam-style multi-step answer generation
- FastAPI endpoint for question answering
- Evaluation scripts for retrieval, QA quality, and hallucination detection
- Critic log and dataset generation for future training

## Tech Stack

- Python
- FastAPI
- Pinecone
- Ollama
- YandexGPT / Yandex embeddings
- Pandas
- pdf2image
- PIL
- PyMuPDF
- pytesseract
- transformers
- torch

## Current Model Setup

This project is currently configured to use:

- LLM provider: `ollama`
- LLM model: `llama3.1:8b`
- Embedding provider: `ollama`
- Embedding model: `nomic-embed-text`
- Critic mode: `teacher`

The code also supports Yandex-based generation/embedding and local critic mode.

## Project Structure

- `rag.py` - main self-refine RAG pipeline
- `critic.py` - critic scoring for retrieval, relevance, grounding, and utility
- `retrieve.py` - metadata-aware retrieval from Pinecone
- `yandex_client.py` - generation backend abstraction
- `yandex_embed.py` - embedding backend abstraction
- `load_data.py` - PDF loading, extraction, OCR fallback, metadata attachment
- `ingest.py` - embedding and Pinecone upload pipeline
- `api.py` - FastAPI `/ask` endpoint
- `evaluate_local_benchmark.py` - retrieval and QA benchmark evaluation
- `evaluate_qa_hallucination_benchmark.py` - hallucination benchmark evaluation
- `critic_data_logger.py` - logs current runs for critic training
- `critic_dataset_builder.py` - builds critic training data

## How It Works

1. Load judgment PDFs and metadata from the dataset.
2. Extract text using native PDF parsing or OCR fallback.
3. Split text into overlapping chunks.
4. Generate embeddings and store them in Pinecone.
5. Retrieve relevant chunks for a query.
6. Use a critic to judge retrieval need, evidence relevance, answer grounding, and utility.
7. Refine answers through adaptive retrieval, post-checking, retry, and beam search.
8. Return the final answer with supporting sources.

## Example Use Cases

- Answer legal questions over court judgments
- Improve grounding in legal question answering
- Reduce hallucinations in retrieval-augmented generation
- Compare baseline RAG against critic-guided self-refining RAG
- Generate data for critic fine-tuning

## Environment Variables

Typical configuration includes:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_DIMENSION`
- `PDF_FOLDER`
- `CSV_PATH`
- `LLM_PROVIDER`
- `OLLAMA_MODEL`
- `OLLAMA_URL`
- `EMBED_PROVIDER`
- `OLLAMA_EMBED_MODEL`
- `USE_CRITIC`
- `CRITIC_MODE`
- `SELF_RAG_MODE`

## Run

Ingest documents:

```bash
python ingest.py

## Start the API
uvicorn api:app --reload

## Ask a question through
POST /ask
