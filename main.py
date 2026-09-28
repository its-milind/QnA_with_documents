import os
import shutil
import tempfile
import time
import asyncio
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY', '')
os.environ['HUGGINGFACE_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN', '')

app = FastAPI(title="Multi-User Isolated RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files and Root Route
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Lazy Shared Embedding Model
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return embedding_model

# Per-User Session Object
class UserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.vector_store: Optional[FAISS] = None
        self.active_filename: Optional[str] = None
        self.chunk_count: int = 0
        self.page_count: int = 0
        self.last_active: float = time.time()

    def update_activity(self):
        self.last_active = time.time()

# In-Memory Dictionary Store for User Sessions
sessions: Dict[str, UserSession] = {}

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class TerminateRequest(BaseModel):
    session_id: str

def get_or_create_session(session_id: Optional[str]) -> UserSession:
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is missing.")
    if session_id not in sessions:
        sessions[session_id] = UserSession(session_id)
    session = sessions[session_id]
    session.update_activity()
    return session

def load_file(file_path: str, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError('Unsupported format! Upload .txt, .docx, or .pdf')

# API Endpoints

@app.get("/api/status")
def get_status(x_session_id: Optional[str] = Header(None)):
    if not x_session_id or x_session_id not in sessions:
        return {"is_loaded": False, "filename": None, "chunk_count": 0, "page_count": 0}

    session = sessions[x_session_id]
    return {
        "is_loaded": session.vector_store is not None,
        "filename": session.active_filename,
        "chunk_count": session.chunk_count,
        "page_count": session.page_count
    }

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None)
):
    session = get_or_create_session(x_session_id)
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ['.pdf', '.txt', '.docx']:
        raise HTTPException(status_code=400, detail="Supported formats: PDF, TXT, DOCX")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        docs = load_file(temp_path, filename)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Build isolated vector store for this user session
        vector_store = FAISS.from_documents(chunks, get_embedding_model())

        session.vector_store = vector_store
        session.active_filename = filename
        session.chunk_count = len(chunks)
        session.page_count = len(docs)

        return {
            "message": f"Successfully loaded {filename}",
            "filename": filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/query")
def ask_question(request: QueryRequest, x_session_id: Optional[str] = Header(None)):
    session = get_or_create_session(x_session_id)

    if session.vector_store is None:
        raise HTTPException(status_code=400, detail="No document indexed for your session.")

    try:
        retriever = session.vector_store.as_retriever(search_kwargs={"k": request.k})
        retrieved_docs = retriever.invoke(request.query)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = SystemMessage(
            content=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know and there is no information in the document.\n\n"
                f"Context:\n{context}"
            )
        )

        llm = init_chat_model(model='openai/gpt-oss-120b', model_provider='groq')
        response = llm.invoke([system_message, HumanMessage(content=request.query)])

        sources = [{"page_content": d.page_content, "metadata": d.metadata} for d in retrieved_docs]
        return {"query": request.query, "answer": response.content, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/terminate-session")
def terminate_session(payload: TerminateRequest):
    """Clean up memory when user leaves or closes tab."""
    session_id = payload.session_id
    if session_id in sessions:
        del sessions[session_id]
        print(f"[Session Cleaned] Removed session: {session_id}")
    return {"status": "terminated"}

@app.delete("/api/clear")
def clear_document(x_session_id: Optional[str] = Header(None)):
    session = get_or_create_session(x_session_id)
    session.vector_store = None
    session.active_filename = None
    session.chunk_count = 0
    session.page_count = 0
    return {"message": "Active index cleared."}

# Background Garbage Collection for Stale Sessions
@app.on_event("startup")
async def start_session_cleanup_task():
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            now = time.time()
            stale_keys = [
                sid for sid, sess in sessions.items()
                if (now - sess.last_active) > 1800  # 30 mins idle timeout
            ]
            for sid in stale_keys:
                del sessions[sid]
                print(f"[Garbage Collector] Evicted idle session: {sid}")

    asyncio.create_task(cleanup_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)