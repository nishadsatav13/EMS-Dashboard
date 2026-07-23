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
# Structured Output Schema (Hardened against LLM formatting errors)
# ============================================================================
class BESSAdvisorySchema(BaseModel):
    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(
        description="Calculated severity of the fault."
    )
    matched_component: str = Field(
        description="Detected ABB component (e.g., Battery, PCS, Transformer, Switchgear, Transmission Line)."
    )
    risk_analysis: str = Field(
        description="Technical explanation of the fault and its risks."
    )
    actions_required: List[str] = Field(
        description="MUST be an array of individual strings (e.g., ['Step 1', 'Step 2']). Do not output a single continuous string."
    )

# ============================================================================
# Connect to Groq — strictly using structured output
# ============================================================================
llm = None
structured_llm = None

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=groq_api_key,
    temperature=0.0,
)

structured_llm = llm.with_structured_output(BESSAdvisorySchema)

# ============================================================================
# Vector Database
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
# RAG Function
# ============================================================================
def generate_rag_advisory(telemetry_alert: str) -> BESSAdvisorySchema:

    # No key — return clear error
    if not groq_api_key:
        return BESSAdvisorySchema(
            severity="WARNING",
            matched_component="Configuration Error",
            risk_analysis="GROQ_API_KEY not found. Add it to Streamlit Cloud Secrets.",
            actions_required=["Go to Streamlit Cloud -> Settings -> Secrets -> add GROQ_API_KEY"]
        )

    # No LLM initialized
    if structured_llm is None:
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
                """You are the NeoAI Expert Agent for an industrial power plant.

Use ONLY the retrieved OEM manual context to answer. 
If the context does not contain the answer, provide a safe, generic industrial isolation protocol.

Rules:
1. Determine the severity (INFO, WARNING, or CRITICAL).
2. Identify the matched_component.
3. Provide a clear, technical risk_analysis.
4. IMPORTANT: actions_required MUST be a JSON array of individual strings. Never output a single continuous string.

Do not include markdown code blocks. Output exactly the requested schema.

OEM MANUAL CONTEXT:
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

        # 1. Build the pipeline with the structured LLM
        pipeline = prompt | structured_llm
        
        # 2. Invoke it
        advisory = pipeline.invoke({
            "context": context_text,
            "alert": telemetry_alert
        })

        print("[NeoAI] Structured JSON generated successfully.")
        
        # 3. Return the fully populated Pydantic object
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
    Fault: thermal_runaway_warning
    Telemetry:
    Temperature = 62.99°C
    SOC = 51.53%
    Voltage = 1197.58V
    """
    result = generate_rag_advisory(sample_alert)
    print(json.dumps(result.model_dump(), indent=4))
