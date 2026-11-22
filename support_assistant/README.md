
# Zepto AI Customer Support Assistant

## 1. Project Overview

The Zepto AI Customer Support Assistant is an intelligent question-answering system designed to answer customer queries using Zepto policy documents.

The application uses Retrieval-Augmented Generation (RAG), sentence embeddings, ChromaDB, LangGraph and FastAPI.

It retrieves relevant information from eight policy documents and provides structured answers with source references and confidence scores.

## 2. Project Objectives

- Answer customer questions about Zepto services.
- Retrieve relevant information from policy documents.
- Reduce repetitive customer support queries.
- Provide accurate and source-based responses.
- Support offline demonstration using mock mode.
- Provide an API for customer interaction.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| LangGraph | AI workflow orchestration |
| FastAPI | REST API |
| Pydantic | Structured response validation |
| Docker | Containerization |
| Groq | Optional real LLM integration |

## 4. Policy Documents

The assistant uses eight policy documents:

1. Delivery Policy
2. Returns and Refunds
3. Membership Tiers
4. Order Tracking
5. Order Cancellation Policy
6. Damaged or Missing Items
7. Gift Cards
8. Customer Support Hours

All policy documents are stored in the `docs/` directory.

## 5. System Architecture

The system follows this workflow:

Customer Question
       |
       v
FastAPI /ask Endpoint
       |
       v
LangGraph Workflow
       |
       v
Intent Classification
       |
       +----------------------+
       |                      |
       v                      v
Policy Question         General Question
       |                      |
       v                      v
Retrieve and Answer      Direct Answer
       |
       v
Sentence Transformer Embeddings
       |
       v
ChromaDB Vector Search
       |
       v
Top 3 Relevant Policy Chunks
       |
       v
Structured Response
       |
       v
Answer + Sources + Confidence

## 6. Document Ingestion and Embedding

The `ingest.py` file performs the following operations:

1. Reads all eight policy documents.
2. Splits documents into smaller overlapping text chunks.
3. Loads the all-MiniLM-L6-v2 sentence-transformer model.
4. Generates vector embeddings for each chunk.
5. Stores text, embeddings and metadata in ChromaDB.
6. Uses cosine distance for similarity-based retrieval.

The database is persisted locally in the `chroma_db/` directory.

## 7. LangGraph Workflow

The assistant contains three main nodes.

### Node 1: classify_intent

Identifies whether a customer question is related to Zepto policies using keyword-based classification.

### Node 2: retrieve_and_answer

- Converts the customer question into an embedding.
- Searches ChromaDB.
- Retrieves the three most relevant policy chunks.
- Generates an answer using the retrieved information.
- Returns source filenames and a confidence score.

### Node 3: direct_answer

Handles general questions that do not require policy retrieval.

Conditional routing determines which node executes after intent classification.

## 8. Structured Prompt Design

The optional real LLM prompt follows five sections:

- ROLE: Defines the assistant as a Zepto customer support assistant.
- CONTEXT: Supplies retrieved policy information.
- TASK: Instructs the model to answer the customer's question.
- FORMAT: Requires answer, sources and confidence.
- LENGTH: Requests a concise customer-friendly response.

An explicit negative constraint prevents the model from inventing policies, prices, refund periods or unsupported promises.

A few-shot example demonstrates the expected response format.

## 9. Mock and Real LLM Modes

### Mock Mode

Mock mode is enabled by default.

Set:

`MOCK_LLM=1`

In this mode:
- Intent classification uses keyword matching.
- Policy retrieval uses real sentence embeddings and ChromaDB.
- The answer is generated from the highest-ranked retrieved policy chunk.
- General questions receive a predefined response.
- No paid LLM API is required.

### Real LLM Mode

Set:

`MOCK_LLM=0`

Configure the `GROQ_API_KEY` environment variable to enable optional Groq integration.

The real LLM uses structured output validation and retries invalid responses up to two additional times.

## 10. API Usage

### Start the application

Run the following command from the `support_assistant` directory after installing dependencies and generating the vector database:

`uvicorn main:app --reload`

The application runs at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Endpoint

`POST /ask`

Example request:

```json
{
  "question": "How long do refunds take?"
}
```

Example response format:

```json
{
  "answer": "Based on the retrieved context: Approved refunds are credited to the original payment method within 3–5 business days...",
  "sources": [
    "doc_02_returns_refunds.txt"
  ],
  "confidence": 0.85
}
```

The confidence value shown is illustrative. Actual values are calculated from retrieval similarity.

### Second example request

```json
{
  "question": "Can I cancel an order after it is packed?"
}
```

The actual response is generated from the retrieved cancellation policy.

## 11. Docker Deployment

Build the Docker image from inside the `support_assistant` directory:

`docker build -t zepto-support-assistant .`

Run the container:

`docker run -p 8000:8000 zepto-support-assistant`

The ChromaDB database must be generated before starting the application. For persistent storage, mount the database directory into the container.

## 12. Project Files

```text
support_assistant/
|
|-- docs/
|   |-- doc_01_delivery_policy.txt
|   |-- doc_02_returns_refunds.txt
|   |-- doc_03_membership_tiers.txt
|   |-- doc_04_order_tracking.txt
|   |-- doc_05_order_cancellation_policy.txt
|   |-- doc_06_damaged_or_missing_items.txt
|   |-- doc_07_gift_cards.txt
|   |-- doc_08_customer_support_hours.txt
|
|-- ingest.py
|-- schemas.py
|-- graph.py
|-- main.py
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- chroma_db/

```

## 13. Conclusion

The Zepto AI Customer Support Assistant combines document retrieval, semantic embeddings, vector search and workflow orchestration to provide policy-based customer support.

Its modular architecture supports offline demonstration through mock mode and optional integration with a real language model.

The application can be extended with additional policy documents, improved intent classification and further customer support features.
