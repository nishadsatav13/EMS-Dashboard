import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Point to your specific PDF
pdf_path = "ABB DOC.pdf"
print(f"[INGEST] Reading engineering manual: {pdf_path}...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Chunking: Split dense pages into searchable paragraphs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"[INGEST] Split document into {len(chunks)} technical chunks.")

# 3. Use a free, highly rated local embedding model (Runs offline on your CPU)
print("[INGEST] Initializing HuggingFace embedding engine...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the Local Vector Database directory
db_storage_path = "./bess_vector_db"
print(f"[INGEST] Building local Vector DB index inside '{db_storage_path}'...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_storage_path
)

print("✅ SUCCESS: Your industrial engineering knowledge base is compiled and vectorized!")