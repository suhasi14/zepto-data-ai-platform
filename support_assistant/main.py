
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import ask_assistant
from schemas import AssistantResponse


# Create FastAPI application
app = FastAPI(
    title="Zepto AI Customer Support Assistant",
    description=(
        "An AI-powered customer support assistant "
        "using LangGraph, ChromaDB and sentence-transformers."
    ),
    version="1.0.0"
)


# Request model
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Customer question about Zepto services."
    )


# Welcome endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the Zepto AI Support Assistant!",
        "status": "running",
        "documentation": "/docs"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Main question-answer endpoint
@app.post(
    "/ask",
    response_model=AssistantResponse
)
def ask_question(request: AskRequest):
    try:
        response = ask_assistant(request.question)

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to process question: {str(error)}"
        )
