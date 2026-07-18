import os
import json
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Load Environment Variables
# ============================================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "❌ GROQ_API_KEY not found. Create a .env file and add your Groq API key."
    )


# ============================================================================
# Connect to Groq LLM
# ============================================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.0
)


# ============================================================================
# Structured Output Schema
# ============================================================================

class BESSAdvisorySchema(BaseModel):

    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(
        description="Calculated severity based on the OEM manual."
    )

    matched_component: str = Field(
        description="Detected hardware component."
    )

    risk_analysis: str = Field(
        description="Technical explanation of the detected problem."
    )

    actions_required: List[str] = Field(
        description="Immediate operator actions."
    )


structured_llm = llm.with_structured_output(BESSAdvisorySchema)


# ============================================================================
# Connect to Local Chroma Vector Database
# ============================================================================

print("[SYSTEM] Connecting to Chroma Vector Database...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


from pathlib import Path

BASE_DIR = Path(__file__).parent

vector_db = Chroma(
    persist_directory=str(BASE_DIR / "bess_vector_db"),
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================================
# RAG Advisory Function
# ============================================================================

def generate_rag_advisory(telemetry_alert: str) -> BESSAdvisorySchema:

    print(f"[SYSTEM] Searching OEM Manual...\n")

    retrieved_docs = retriever.invoke(telemetry_alert)

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the ABB BESS Expert Agent for the NeoAI EMS Dashboard.

Your task is to analyze incoming telemetry using ONLY the retrieved OEM manual.

Rules:

1. Never invent information.
2. Base every conclusion only on the retrieved manual.
3. Identify the affected ABB component.
4. Assess severity.
5. Explain the technical risk.
6. Provide clear operator actions.

OEM MANUAL:

{context}
"""
            ),

            (
                "user",
                """
Incoming Telemetry:

{alert}
"""
            )
        ]
    )

    pipeline = prompt | structured_llm

    try:

        return pipeline.invoke(
            {
                "context": context_text,
                "alert": telemetry_alert
            }
        )

    except Exception as e:

        print(e)

        return BESSAdvisorySchema(

            severity="CRITICAL",

            matched_component="Unknown",

            risk_analysis="Unable to generate advisory because the AI pipeline failed.",

            actions_required=[
                "Inspect system logs.",
                "Verify Groq API connectivity.",
                "Review retrieved OEM documentation manually."
            ]
        )


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":

    print("\n========== NeoAI RAG Test ==========\n")

    sample_alert = """
Component: PCS

Fault: Arc Flash

Telemetry:

Light detected inside PCS enclosure.
Current exceeded threshold.
Temperature increasing rapidly.
"""

    result = generate_rag_advisory(sample_alert)

    print(json.dumps(result.model_dump(), indent=4))