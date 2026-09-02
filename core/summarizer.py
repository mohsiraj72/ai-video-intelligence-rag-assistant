from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
from core.logger_config import get_logger

logger = get_logger(__name__)

load_dotenv()  # Load environment variables from .env file

# Configuration constants
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
TITLE_CONTEXT_LENGTH = 2000
TEMPERATURE = 0.3 

def get_llm():
    """Initialize and return a Mistral AI language model instance.
    
    Returns:
        ChatMistralAI: Configured LLM instance with API key from environment.
        
    Raises:
        ValueError: If MISTRAL_API_KEY is not set in environment.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=TEMPERATURE
    )


def split_transcript(transcript: str) -> list:
    """Split transcript into manageable chunks for processing.
    
    Args:
        transcript: The full transcript text to split.
        
    Returns:
        list: List of text chunks.
        
    Raises:
        ValueError: If transcript is empty.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    """Generate a professional summary of a meeting transcript using map-reduce approach.
    
    Args:
        transcript: The full meeting transcript to summarize.
        
    Returns:
        str: A professional bullet-point summary of the meeting.
        
    Raises:
        ValueError: If transcript is empty.
        Exception: If API call fails.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Starting summarization using map-reduce approach...")
    try:
        llm = get_llm()

        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ])

        map_chain = map_prompt | llm | StrOutputParser()

        chunks = split_transcript(transcript)
        logger.info(f"Summarizing {len(chunks)} chunks...")
        chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
        combined = "\n\n".join(chunk_summaries)

        combined_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ])

        combined_chain = (
            RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
        )

        result = combined_chain.invoke(combined)
        logger.info("Summarization completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error summarizing transcript: {str(e)}", exc_info=True)
        raise Exception(f"Error summarizing transcript: {str(e)}")

def generate_title(transcript: str) -> str:
    """Generate a concise professional title for a meeting transcript.
    
    Args:
        transcript: The meeting transcript text.
        
    Returns:
        str: A short professional title (max 8 words).
        
    Raises:
        ValueError: If transcript is empty.
        Exception: If API call fails.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Generating meeting title...")
    try:
        llm = get_llm()

        title_chain = (
            RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | 
            ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ])
            | llm
            | StrOutputParser()
        )

        result = title_chain.invoke(transcript[:TITLE_CONTEXT_LENGTH])
        logger.info(f"Title generated: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating title: {str(e)}", exc_info=True)
        raise Exception(f"Error generating title: {str(e)}")





