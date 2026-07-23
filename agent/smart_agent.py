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

# ============================================================================
# Connect to Groq — only if key exists
# ============================================================================
llm = None
structured_llm = None

if groq_api_key:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(BESSAdvisorySchema)

# ============================================================================
# Vector Database — only if folder exists
# ============================================================================
retriever = None

BASE_DIR = Path(__file__).parent
VECTOR_PATH = BASE_DIR / "bess_vector_db"

if VECTOR_PATH.exists() and groq_api_key:
    print("[NeoAI] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=str(VECTOR_PATH),
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    print("[NeoAI] Chroma connected successfully.")
else:
    print(f"[NeoAI] WARNING: Vector DB not found at {VECTOR_PATH} or GROQ key missing.")

# ============================================================================
# RAG Function — same as original, just safe
# ============================================================================
def generate_rag_advisory(telemetry_alert: str) -> BESSAdvisorySchema:

    # No key — return clear error
    if not groq_api_key:
        return BESSAdvisorySchema(
            severity="WARNING",
            matched_component="Configuration Error",
            risk_analysis="GROQ_API_KEY not found. Add it to Streamlit Cloud Secrets.",
            actions_required=["Go to Streamlit Cloud → Settings → Secrets → add GROQ_API_KEY"]
        )

    # No LLM initialized
    if llm is None:
        return BESSAdvisorySchema(
            severity="WARNING",
            matched_component="LLM Error",
            risk_analysis="LLM failed to initialize.",
            actions_required=["Check Streamlit logs for details."]
        )

    try:
        print("\n==============================")
        print("[NeoAI] Incoming Alert")
        print(telemetry_alert)
        print("==============================")

        # Get context from vector DB if available
        context_text = "No OEM manual context available — answering from model knowledge."
        if retriever is not None:
            retrieved_docs = retriever.invoke(telemetry_alert)
            print(f"[NeoAI] Retrieved {len(retrieved_docs)} document(s).")
            if retrieved_docs:
                context_text = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are the ABB BESS Expert Agent.

Use ONLY the retrieved OEM manual.

Rules:
1. Never hallucinate.
2. Never invent information.
3. Base every answer only on the OEM manual.
4. Identify component.
5. Assess severity.
6. Explain risk.
7. Return ONLY plain English.
8. Do NOT use tool calling.
9. Do NOT return JSON.
10. Do NOT use function calling.

OEM MANUAL:
{context}
"""
            ),
            (
                "user",
                """Incoming Telemetry:
{alert}
"""
            )
        ])

        # Use the structured_llm so it dynamically populates the Pydantic schema!
        pipeline = prompt | structured_llm
        
        advisory = pipeline.invoke({
            "context": context_text,
            "alert": telemetry_alert
        })

        print("[NeoAI] Structured output generated successfully!")
        
        return advisory

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
