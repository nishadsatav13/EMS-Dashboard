import os
import json
import traceback
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Structured Output Schema
# ============================================================================

class BESSAdvisorySchema(BaseModel):
    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(
        description="Calculated severity."
    )
    matched_component: str = Field(
        description="Detected BESS component."
    )
    risk_analysis: str = Field(
        description="Technical explanation."
    )
    actions_required: List[str] = Field(
        description="Operator actions."
    )


# ============================================================================
# Lazy initialization — nothing runs at import time
# This prevents Streamlit Cloud crash on startup
# ============================================================================

_llm = None
_retriever = None
_initialized = False
_init_error = None


def _initialize():
    """
    Initialize LLM + vector DB only when first called.
    Not at import time — so dashboard loads even if keys/files are missing.
    """
    global _llm, _retriever, _initialized, _init_error

    if _initialized:
        return

    try:
        from dotenv import load_dotenv
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to Streamlit Cloud Secrets: "
                "Settings → Secrets → GROQ_API_KEY = 'your-key-here'"
            )

        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.0,
        )

        # Vector DB — only load if it exists
        BASE_DIR = Path(__file__).parent
        VECTOR_PATH = BASE_DIR / "bess_vector_db"

        if VECTOR_PATH.exists():
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            print("[NeoAI] Loading embedding model (this takes ~30s first time)...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            vector_db = Chroma(
                persist_directory=str(VECTOR_PATH),
                embedding_function=embeddings
            )
            _retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            print("[NeoAI] Chroma vector DB connected.")
        else:
            print(f"[NeoAI] WARNING: Vector DB not found at {VECTOR_PATH}. "
                  "RAG context will be skipped — LLM will answer from training only.")
            _retriever = None

        _initialized = True
        print("[NeoAI] Agent initialized successfully.")

    except Exception as e:
        _init_error = str(e)
        _initialized = True  # mark as attempted so we don't retry every call
        print(f"[NeoAI] Initialization error: {e}")


# ============================================================================
# RAG Advisory Function
# ============================================================================

def generate_rag_advisory(telemetry_alert: str) -> BESSAdvisorySchema:
    """
    Generate a RAG-based advisory for a BESS telemetry alert.
    Safe to call from dashboard — will never crash the app.
    """

    # Initialize on first call
    _initialize()

    # If init failed, return a clear error advisory
    if _init_error:
        return BESSAdvisorySchema(
            severity="WARNING",
            matched_component="Agent Initialization",
            risk_analysis=(
                f"Agent could not initialize: {_init_error}. "
                "Check GROQ_API_KEY in Streamlit Secrets and ensure "
                "bess_vector_db folder is in your GitHub repo."
            ),
            actions_required=[
                "Go to Streamlit Cloud → Settings → Secrets",
                "Add GROQ_API_KEY = 'your-groq-key'",
                "Make sure bess_vector_db/ folder is committed to GitHub",
                "Redeploy the app"
            ]
        )

    try:
        from langchain_core.prompts import ChatPromptTemplate

        # Retrieve context from vector DB if available
        context_text = "No OEM manual context available."
        if _retriever is not None:
            retrieved_docs = _retriever.invoke(telemetry_alert)
            if retrieved_docs:
                context_text = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )
                print(f"[NeoAI] Retrieved {len(retrieved_docs)} document(s).")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are NeoAI — a BESS (Battery Energy Storage System) expert agent.

Use the OEM manual context below to ground your answer.
If context is unavailable, use your technical knowledge.

Rules:
1. Never hallucinate. If unsure, say so.
2. Identify the component and fault clearly.
3. Assess severity: INFO / WARNING / CRITICAL.
4. Explain the risk in plain technical English.
5. Give specific actionable steps for the operator.

OEM MANUAL CONTEXT:
{context}
"""
            ),
            (
                "user",
                """Incoming BESS Telemetry Alert:

{alert}

Provide your analysis."""
            )
        ])

        pipeline = prompt | _llm
        response = pipeline.invoke({
            "context": context_text,
            "alert": telemetry_alert
        })

        # Parse severity from response text
        content = response.content.upper()
        if "CRITICAL" in content:
            severity = "CRITICAL"
        elif "WARNING" in content or "WARN" in content:
            severity = "WARNING"
        else:
            severity = "INFO"

        return BESSAdvisorySchema(
            severity=severity,
            matched_component=_extract_component(telemetry_alert),
            risk_analysis=response.content,
            actions_required=["Review agent analysis above and take appropriate action."]
        )

    except Exception as e:
        traceback.print_exc()
        return BESSAdvisorySchema(
            severity="CRITICAL",
            matched_component=f"ERROR: {type(e).__name__}",
            risk_analysis=str(e),
            actions_required=[
                "Check Streamlit app logs for full traceback.",
                "Verify GROQ_API_KEY is valid and has credits.",
                "Check internet connectivity from Streamlit Cloud."
            ]
        )


def _extract_component(alert_text: str) -> str:
    """Extract component name from alert text."""
    alert_lower = alert_text.lower()
    for comp in ["battery", "pcs", "transformer", "switchgear", "transmission"]:
        if comp in alert_lower:
            return comp.title()
    return "BESS System"


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
