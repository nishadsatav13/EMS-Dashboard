import os
import json
import traceback
from pathlib import Path
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
        "❌ GROQ_API_KEY not found. Configure it in Streamlit Secrets or .env"
    )


# ============================================================================
# Connect to Groq
# ============================================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.0,
)


# ============================================================================
# Structured Output
# ============================================================================

class BESSAdvisorySchema(BaseModel):

    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(
        description="Calculated severity."
    )

    matched_component: str = Field(
        description="Detected ABB component."
    )

    risk_analysis: str = Field(
        description="Technical explanation."
    )

    actions_required: List[str] = Field(
        description="Operator actions."
    )


structured_llm = llm


# ============================================================================
# Vector Database
# ============================================================================

print("[NeoAI] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

BASE_DIR = Path(__file__).parent

VECTOR_PATH = BASE_DIR / "bess_vector_db"

print(f"[NeoAI] Vector DB path: {VECTOR_PATH}")

if not VECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Vector database not found at {VECTOR_PATH}"
    )

vector_db = Chroma(
    persist_directory=str(VECTOR_PATH),
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

print("[NeoAI] Chroma connected successfully.")


# ============================================================================
# RAG Function
# ============================================================================

def generate_rag_advisory(
    telemetry_alert: str
) -> BESSAdvisorySchema:

    try:

        print("\n==============================")
        print("[NeoAI] Incoming Alert")
        print(telemetry_alert)
        print("==============================")

        retrieved_docs = retriever.invoke(telemetry_alert)

        print(f"[NeoAI] Retrieved {len(retrieved_docs)} document(s).")

        context_text = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the ABB BESS Expert Agent.

Use ONLY the retrieved OEM manual.

Rules:

1. Never hallucinate.
2. Never invent information.
3. Base every answer only on the OEM manual.
4. Identify component.
5. Assess severity.
6. Explain risk.
7.Return ONLY plain English.
8.Do NOT use tool calling.
9.Do NOT return JSON.
10.Do NOT use function calling.

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

response = pipeline.invoke(
    {
        "context": context_text,
        "alert": telemetry_alert
    }
)

print(response.content)

return BESSAdvisorySchema(
    severity="INFO",
    matched_component="TEST",
    risk_analysis=response.content,
    actions_required=["Generated successfully"]
)

    except Exception as e:

        print("\n==============================")
        print("[NeoAI ERROR]")
        traceback.print_exc()
        print("==============================")

        return BESSAdvisorySchema(

            severity="CRITICAL",

            matched_component=f"ERROR: {type(e).__name__}",

            risk_analysis=str(e),

            actions_required=[
                "Open Streamlit logs.",
                "Inspect traceback above.",
                "Resolve the reported exception."
            ]
        )


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":

    sample_alert = """
Component: Battery

Fault: Cell Over Temperature

Telemetry:

Temperature = 72°C
SOC = 91%
Voltage = 810V
"""

    result = generate_rag_advisory(sample_alert)

    print(json.dumps(result.model_dump(), indent=4))
