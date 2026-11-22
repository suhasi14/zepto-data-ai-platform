
import os
from pathlib import Path
from typing import TypedDict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from schemas import AssistantResponse

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"

# Mock mode is enabled by default.
# Set MOCK_LLM=0 to enable optional real LLM mode.
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

_embedding_model = None
_chroma_collection = None


class AssistantState(TypedDict, total=False):
    question: str
    intent: str
    retrieved_chunks: List[str]
    sources: List[str]
    answer: str
    confidence: float
    response: dict


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def get_collection():
    global _chroma_collection

    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=str(DB_DIR)
        )

        _chroma_collection = client.get_collection(
            name="zepto_policies"
        )

    return _chroma_collection


def classify_intent(state: AssistantState):
    """Identify whether the question concerns a Zepto policy."""

    question = state["question"].lower()

    policy_keywords = [
        "delivery",
        "deliver",
        "refund",
        "return",
        "cancel",
        "cancellation",
        "tracking",
        "track",
        "membership",
        "pass",
        "gift card",
        "giftcard",
        "damaged",
        "spoiled",
        "missing",
        "replacement",
        "support hours",
        "customer support",
        "delivery fee",
        "priority delivery"
    ]

    if any(keyword in question for keyword in policy_keywords):
        intent = "policy"
    else:
        intent = "general"

    return {"intent": intent}


# --------------------------------------------------
# CONDITIONAL ROUTING
# --------------------------------------------------

def route_intent(state: AssistantState):
    """Route policy questions to retrieval and others to direct answer."""

    if state["intent"] == "policy":
        return "retrieve_and_answer"

    return "direct_answer"


def retrieve_policy(question: str):
    """Retrieve the three most relevant policy chunks."""

    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    sources = []

    for metadata in metadatas:
        source = metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)

    # Convert cosine distance into a bounded confidence estimate.
    if distances:
        confidence = max(
            0.0,
            min(1.0, 1.0 - float(distances[0]))
        )
    else:
        confidence = 0.0

    return chunks, sources, confidence


def build_prompt(question: str, context: str):
    """Build a structured prompt for the optional real LLM."""

    return f"""
ROLE:
You are a professional Zepto customer support assistant.

CONTEXT:
Use only the following retrieved Zepto policy information:
{context}

TASK:
Answer the customer's question accurately using the supplied context.

FORMAT:
Return a structured response containing:
- answer: a clear and helpful answer.
- sources: the policy document filenames used.
- confidence: a number between 0 and 1.

LENGTH:
Keep the answer concise and easy for a customer to understand.

NEGATIVE CONSTRAINT:
Do not invent policies, prices, delivery times, refund periods,
or promises that are not present in the supplied context.
If the context does not contain the answer, clearly say that
the available policy information does not specify it.

FEW-SHOT EXAMPLE:

Customer question:
How long do refunds take?

Retrieved context:
Approved refunds are credited to the original payment method
within 3–5 business days.

Expected answer:
Approved refunds reach the original payment method within
3–5 business days.

Customer question:
{question}

Answer using the retrieved context only.
"""


# --------------------------------------------------
# OPTIONAL REAL LLM
# --------------------------------------------------

def generate_real_answer(question, context, sources):
    """Generate a structured answer using optional Groq LLM."""

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

    structured_llm = llm.with_structured_output(
        AssistantResponse
    )

    prompt = build_prompt(question, context)

    # Initial attempt plus two retries for invalid output.
    last_error = None

    for attempt in range(3):
        try:
            result = structured_llm.invoke(prompt)

            if isinstance(result, AssistantResponse):
                response = result
            else:
                response = AssistantResponse.model_validate(result)

            if not response.sources:
                response.sources = sources

            return response

        except Exception as error:
            last_error = error
            prompt += (
                "\nYour previous output was invalid. "
                "Return valid structured output matching the schema."
            )

    raise ValueError(
        f"Could not generate valid structured output: {last_error}"
    )


# --------------------------------------------------
# NODE 2: RETRIEVE AND ANSWER
# --------------------------------------------------

def retrieve_and_answer(state: AssistantState):
    """Retrieve policy information and prepare the response."""

    question = state["question"]

    chunks, sources, confidence = retrieve_policy(question)

    if not chunks:
        answer = (
            "I could not find relevant information in the "
            "available Zepto policies. Please contact Zepto support."
        )

        return {
            "retrieved_chunks": [],
            "sources": [],
            "answer": answer,
            "confidence": 0.0,
            "response": {
                "answer": answer,
                "sources": [],
                "confidence": 0.0
            }
        }

    # Mock mode uses actual retrieved policy text.
    if MOCK_LLM:
        answer = (
            "Based on the retrieved context: "
            + chunks[0]
        )

        response = AssistantResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )

    else:
        context = "\n\n".join(chunks)

        try:
            response = generate_real_answer(
                question,
                context,
                sources
            )

        except Exception:
            # Keep the application usable if optional LLM mode fails.
            answer = (
                "Based on the retrieved context: "
                + chunks[0]
            )

            response = AssistantResponse(
                answer=answer,
                sources=sources,
                confidence=confidence
            )

    return {
        "retrieved_chunks": chunks,
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
        "response": response.model_dump()
    }

def direct_answer(state: AssistantState):
    """Handle general questions without policy retrieval."""

    answer = (
        "Hello! I am Zepto's AI support assistant. "
        "I can help with delivery, returns, refunds, "
        "orders, memberships, gift cards, and customer support. "
        "What would you like to know?"
    )

    response = AssistantResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )

    return {
        "retrieved_chunks": [],
        "sources": [],
        "answer": answer,
        "confidence": 1.0,
        "response": response.model_dump()
    }


def build_assistant_graph():
    """Create and compile the three-node LangGraph workflow."""

    workflow = StateGraph(AssistantState)

    workflow.add_node(
        "classify_intent",
        classify_intent
    )

    workflow.add_node(
        "retrieve_and_answer",
        retrieve_and_answer
    )

    workflow.add_node(
        "direct_answer",
        direct_answer
    )

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge(
        "retrieve_and_answer",
        END
    )

    workflow.add_edge(
        "direct_answer",
        END
    )

    return workflow.compile()


# Compiled graph used by the application
assistant_graph = build_assistant_graph()


def ask_assistant(question: str):
    """Send a question through the LangGraph workflow."""

    result = assistant_graph.invoke({
        "question": question
    })

    return AssistantResponse.model_validate(
        result["response"]
    )
