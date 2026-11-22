
from pydantic import BaseModel, Field
from typing import List


class AssistantResponse(BaseModel):
    """Structured response returned by the Zepto support assistant."""

    answer: str = Field(
        ...,
        description="Clear answer to the customer's question."
    )

    sources: List[str] = Field(
        default_factory=list,
        description="Names of the policy documents used."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
