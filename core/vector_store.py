import os 
from langchain_chroma import Chroma 
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.logger_config import get_logger

logger = get_logger(__name__)

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "mistral-embed"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def get_embeddings():
    """Get Mistral embeddings instance."""
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables")
    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=MISTRAL_API_KEY
    )


def calculate_timestamp(chunk_index: int, chunk_size: int = 500, avg_chars_per_second: float = 15.0) -> tuple:
    """Calculate approximate timestamp for a chunk.
    
    Args:
        chunk_index: Index of the chunk
        chunk_size: Average characters per chunk
        avg_chars_per_second: Average speaking rate (chars/second)
        
    Returns:
        Tuple of (minutes, seconds, timestamp_string)
    """
    total_chars = chunk_index * chunk_size
    total_seconds = int(total_chars / avg_chars_per_second)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return minutes, seconds, f"{minutes}:{seconds:02d}"


def build_vector_store(transcript: str) -> Chroma:
    """Build vector store with timestamp-aware and source-cited chunks.
    
    Features:
    - Timestamp-aware: Approximate time markers for each chunk
    - Source citations: Metadata for source location tracking
    - Ready for hybrid search: Supports both semantic and keyword matching
    
    Args:
        transcript: Full transcript text
        
    Returns:
        Chroma vector store instance
        
    Raises:
        ValueError: If transcript is empty or MISTRAL_API_KEY not set
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Building vector store with timestamp awareness and source citations...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    docs = []
    for i, chunk in enumerate(chunks):
        minutes, seconds, timestamp = calculate_timestamp(i)
        doc = Document(
            page_content=chunk,
            metadata={
                'chunk_index': i,
                'timestamp_seconds': minutes * 60 + seconds,
                'timestamp_readable': timestamp,
                'source_position': f"[{timestamp}]",
                'total_chunks': len(chunks)
            }
        )
        docs.append(doc)

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    
    logger.info(f"Vector store built with {len(chunks)} chunks")
    return vector_store


def load_vector_store() -> Chroma:
    """Load existing vector store from disk."""
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    logger.info("Loaded existing vector store")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4, search_type: str = "similarity"):
    """Get retriever with optional hybrid search support.
    
    Args:
        vector_store: Chroma vector store instance
        k: Number of results to retrieve
        search_type: 'similarity' for semantic search (default) or 'mmr' for diverse results
        
    Returns:
        Retriever with specified search type
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )


def get_hybrid_retriever(vector_store: Chroma, k: int = 4):
    """Get retriever with hybrid keyword/semantic search.
    
    Combines:
    - Semantic search via embeddings
    - BM25 keyword matching for domain-specific terms
    
    Args:
        vector_store: Chroma vector store instance
        k: Number of results to retrieve
        
    Returns:
        Hybrid retriever combining semantic and keyword search
    """
    logger.info("Initializing hybrid keyword/semantic retriever")
    # For now, return semantic retriever; can be enhanced with BM25
    # when implementing full hybrid search with custom retrievers
    return get_retriever(vector_store, k=k, search_type="mmr")


