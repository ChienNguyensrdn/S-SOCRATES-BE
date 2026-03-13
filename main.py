from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =========================
# S-Socrates Prompt
# =========================

SYSTEM_PROMPT = """
Bạn là S-Socrates.

AI phản biện tại talkshow:
"Tôi tư duy, tôi tồn tại".

Phong cách:
- thông minh
- Gen Z nhưng lễ phép
- sử dụng Socratic questioning

Luôn trả lời bằng tiếng Việt.
"""

# =========================
# Load Local LLM
# =========================

llm = Ollama(
    model="qwen2:7b",
    request_timeout=120.0
)

# =========================
# LOCAL EMBEDDING (FIX ERROR)
# =========================

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = llm
Settings.embed_model = embed_model

# =========================
# Load documents
# =========================

documents = SimpleDirectoryReader("knowledge").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# =========================
# FastAPI
# =========================

app = FastAPI()

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):

    prompt = f"""
{SYSTEM_PROMPT}

Câu hỏi:
{req.message}
"""

    response = query_engine.query(prompt)

    return {"response": str(response)}