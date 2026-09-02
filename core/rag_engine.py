import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever, get_hybrid_retriever
from core.logger_config import get_logger

logger = get_logger(__name__)

def get_llm():
    """Initialize and return a Mistral AI language model instance."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables")
    
    logger.info("Initializing Mistral LLM")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    """Format retrieved documents with source citations and timestamps."""
    formatted = []
    for doc in docs:
        timestamp = doc.metadata.get('timestamp_readable', 'N/A')
        content = f"[{timestamp}] {doc.page_content}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(retriever, llm, prompt):
    """Create a RAG chain with source citations.
    
    Args:
        retriever: Retriever instance
        llm: Language model instance
        prompt: Chat prompt template
        
    Returns:
        Complete RAG chain with citation support
    """
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_rag_chain(transcript: str, use_hybrid_search: bool = True):
    """Build RAG chain with advanced features.
    
    Features:
    - Timestamp-aware retrieval
    - Source citations
    - Hybrid keyword/semantic search
    
    Args:
        transcript: Full transcript text
        use_hybrid_search: Use hybrid search if True, else semantic only
        
    Returns:
        Complete RAG chain ready for queries
        
    Raises:
        ValueError: If transcript is empty
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Building RAG chain...")
    vector_store = build_vector_store(transcript)

    # Use hybrid or semantic retriever
    if use_hybrid_search:
        logger.info("Using hybrid keyword/semantic search")
        retriever = get_hybrid_retriever(vector_store, k=4)
    else:
        logger.info("Using semantic search only")
        retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            """You are an expert meeting assistant with access to transcript excerpts.
Answer the user's question based ONLY on the meeting transcript context provided below.

Each context excerpt includes a timestamp [MM:SS] indicating when it was said.

Guidelines:
- Quote directly from the transcript when possible
- Reference timestamps to indicate when information was mentioned
- If the answer is not found in the context, say: "I could not find this information in the meeting transcript."
- Always be concise and precise
- When citing information, include the timestamp

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = create_rag_chain(retriever, llm, prompt)
    logger.info("RAG chain built successfully")
    return rag_chain


def load_rag_chain(use_hybrid_search: bool = True):
    """Load previously built RAG chain from vector store.
    
    Args:
        use_hybrid_search: Use hybrid search if True
        
    Returns:
        Complete RAG chain
    """
    logger.info("Loading existing RAG chain...")
    vector_store = load_vector_store()
    
    if use_hybrid_search:
        retriever = get_hybrid_retriever(vector_store, k=4)
    else:
        retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant with access to transcript excerpts.
Answer the user's question based ONLY on the meeting transcript context provided below.

Each context excerpt includes a timestamp [MM:SS] indicating when it was said.

Guidelines:
- Quote directly from the transcript when possible
- Reference timestamps to indicate when information was mentioned
- If the answer is not found in the context, say: "I could not find this information in the meeting transcript."
- Always be concise and precise
- When citing information, include the timestamp

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = create_rag_chain(retriever, llm, prompt)
    logger.info("RAG chain loaded successfully")
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """Ask a question using the RAG chain.
    
    Args:
        rag_chain: RAG chain instance
        question: User question
        
    Returns:
        Answer with citations and timestamps
        
    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    logger.info(f"Processing question: {question}")
    try:
        answer = rag_chain.invoke(question)
        logger.info(f"Generated answer of length: {len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise