import os
import uuid
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

class embedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load()

    def _load(self) -> None:
        self.model = SentenceTransformer(self.model_name)

    def generating(self, text: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.encode(text, show_progress_bar=True, convert_to_numpy=True)

class vector_store:
    def __init__(self, collection: str = "sebi_docs", persist: str = "./vectorstore"):
        self.collection_name = collection
        self.persist = persist
        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.batch_size = 5000  # ChromaDB max batch size
        self.initialize()

    def initialize(self) -> None:
        os.makedirs(self.persist, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "pdf documents"}
        )

    def adds(self, documents: List[Any], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents size not equal to embedding")
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        # Process in batches to avoid ChromaDB batch size limit
        total_docs = len(documents)
        st.write(f"📊 Adding {total_docs} chunks in batches...")
        
        for batch_start in range(0, total_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_docs)
            batch_docs = documents[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            
            ids, metadatas, doc_text, embedding_list = [], [], [], []
            for i, (doc, emb) in enumerate(zip(batch_docs, batch_embeddings)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{batch_start + i}"
                ids.append(doc_id)
                metadata = dict(doc.metadata)
                metadata["doc_id"] = batch_start + i
                metadatas.append(metadata)
                doc_text.append(doc.page_content)
                embedding_list.append(emb.tolist())
            
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                embeddings=embedding_list,
                documents=doc_text
            )
            st.write(f"✅ Added batch {batch_start}-{batch_end}/{total_docs}")

    def reset(self) -> None:
        if self.client is None:
            raise ValueError("Client not initialized")
        if self.collection is None:
            raise ValueError("Collection not initialized")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "pdf documents"}
        )

    def count(self) -> int:
        if self.collection is None:
            raise ValueError("Collection not initialized")
        return self.collection.count()


class retrivel:
    def __init__(self, vector_store_obj: vector_store, embedding_obj: embedding):
        self.vector_store = vector_store_obj
        self.embedding_manager = embedding_obj

    def retrive(self, query: str, threshold: float = 0.0, k: int = 5) -> List[Dict[str, Any]]:
        if self.vector_store.collection is None:
            raise ValueError("Vector store collection not initialized")
        
        query_embedding = self.embedding_manager.generating([query])[0]
        result = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        retrived = []
        if result['documents'] and result['documents'][0]:
            documents = result['documents'][0]
            metadata  = result['metadatas'][0]
            distance  = result['distances'][0]
            ids       = result['ids'][0]
            for i, (document_id, meta, dist, document) in enumerate(
                zip(ids, metadata, distance, documents)
            ):
                similarity = 1 - dist
                if similarity >= threshold:
                    retrived.append({
                        'id':         document_id,
                        'content':    document,
                        'metadata':   meta,
                        'distance':   dist,
                        'similarity': round(similarity, 3),
                        'rank':       i + 1
                    })
        return retrived


def convert(pdf_paths: List[str]) -> List[Any]:
    all_doc = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        document = loader.load()
        for doc in document:
            doc.metadata["source_file"] = Path(path).name
        all_doc.extend(document)
    return all_doc


def split(documents: List[Any], size: int = 500, overlap: int = 100) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def build_fallback_answer(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "I couldn't find relevant SEBI content for your question."

    best_match = results[0]
    excerpt = " ".join(str(best_match.get("content", "")).split())
    if not excerpt:
        return "I found relevant SEBI document chunks, but the text was empty."

    excerpt = excerpt[:1200]
    return (
        f"Based on the retrieved SEBI documents, the most relevant excerpt is:\n\n{excerpt}"
    )


def ragg(query: str, retriver_obj: retrivel, llm: HuggingFaceEndpoint, top: int = 5) -> tuple[str, List[Dict[str, Any]]]:
    results = retriver_obj.retrive(query, k=top)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No context found in the documents.", []

    prompt = f"""You are a helpful assistant specializing in SEBI regulations and Indian financial markets.
Answer strictly based on the context below.
If the answer is not in the context, say "I couldn't find this in the SEBI documents."

Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return str(response), results
    except Exception:
        return build_fallback_answer(query, results), results


SEBI_DOCS_FOLDER = "./sebi_docs"
SEBI_PDF_FILES   = ["sebi.pdf", "sebi1.pdf", "sebi2.pdf", "sebi3.pdf", "sebi4.pdf"]


def get_pdf_paths() -> List[str]:
    paths = []
    for fname in SEBI_PDF_FILES:
        full_path = os.path.join(SEBI_DOCS_FOLDER, fname)
        if os.path.exists(full_path):
            paths.append(full_path)
        else:
            st.sidebar.warning(f"⚠️ Missing: {fname}")
    return paths


@st.cache_resource(show_spinner=False)
def load_system() -> tuple[Optional[retrivel], Optional[HuggingFaceEndpoint], Optional[str]]:
    # Get HF API key
    hf_key: Optional[str] = HF_API_KEY
    if not hf_key:
        try:
            hf_key = st.secrets.get("HF_API_KEY")
        except Exception:
            pass

    if not hf_key:
        return None, None, "❌ HF_API_KEY not found. Add it to .env or Streamlit secrets."

    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        return None, None, f"❌ No PDFs found in {SEBI_DOCS_FOLDER}/ folder."

    # Load and chunk documents
    all_docs = convert(pdf_paths)
    chunks = split(all_docs)

    # Generate embeddings
    embed_obj = embedding()
    texts = [doc.page_content for doc in chunks]
    embeddings_arr = embed_obj.generating(texts)

    # Initialize vector store
    vs = vector_store(persist="./vectorstore")
    if vs.count() > 0:
        vs.reset()
    vs.adds(chunks, embeddings_arr)

    # Initialize retriever
    ret = retrivel(vs, embed_obj)

    # Initialize LLM (HuggingFace - free!)
    try:
        llm = HuggingFaceEndpoint(
            model="gpt2",
            huggingfacehub_api_token=hf_key,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        return None, None, f"❌ HuggingFace API error: {str(e)}"

    return ret, llm, None


st.set_page_config(page_title="SEBI RAG Chatbot", page_icon="📄", layout="centered")
st.title("📄 SEBI Document Q&A")
st.caption("Powered by SEBI Master Circulars · Mistral · RAG (100% Free)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "retrivel_obj" not in st.session_state:
    st.session_state.retrivel_obj = None

if "llm" not in st.session_state:
    st.session_state.llm = None

with st.sidebar:
    st.header("📚 Loaded Documents")
    # st.caption("These SEBI circulars are preloaded:")
    # for fname in SEBI_PDF_FILES:
    #     full_path = os.path.join(SEBI_DOCS_FOLDER, fname)
    #     icon = "✅" if os.path.exists(full_path) else "❌"
    #     st.caption(f"{icon} {fname}")

    st.markdown("---")
    st.markdown("**Model:** Mistral 7B (HuggingFace)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Initialize system on first run
if not st.session_state.system_ready:
    with st.spinner("🔄 Loading SEBI documents..."):
        ret, llm, error = load_system()

    if error:
        st.error(error)
        st.stop()

    if ret is None or llm is None:
        st.error("❌ Failed to initialize the retrieval system.")
        st.stop()

    st.session_state.retrivel_obj = ret
    st.session_state.llm = llm
    st.session_state.system_ready = True
    st.success("✅ SEBI documents loaded successfully!")

# Chat input
query = st.chat_input("Ask about SEBI regulations...")

if query:
    retriever_obj = st.session_state.get("retrivel_obj")
    llm_obj = st.session_state.get("llm")
    if retriever_obj is None or llm_obj is None:
        st.error("The retrieval system is not ready yet. Please refresh the page.")
        st.stop()

    with st.chat_message("user"):
        st.write(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching SEBI documents..."):
            answer, sources = ragg(
                query,
                cast(retrivel, retriever_obj),
                cast(HuggingFaceEndpoint, llm_obj),
                top=5
            )
        st.write(answer)

        if sources:
            with st.expander("📎 Source chunks used"):
                for src in sources:
                    st.markdown(
                        f"**Rank {src['rank']}** &nbsp;|&nbsp; "
                        f"File: `{src['metadata'].get('source_file', '?')}` &nbsp;|&nbsp; "
                        f"Page: `{src['metadata'].get('page', '?')}` &nbsp;|&nbsp; "
                        f"Similarity: `{src['similarity']}`"
                    )
                    st.caption(src['content'][:300] + "...")
                    st.markdown("---")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })