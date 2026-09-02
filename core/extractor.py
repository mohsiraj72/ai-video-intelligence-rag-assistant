#Actionable items, decisions, questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from core.logger_config import get_logger

logger = get_logger(__name__) 


def get_llm():
    """Get Mistral LLM instance for extraction tasks."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables")
    
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.2
    )


def build_chain(system_prompt: str):
    """Build extraction chain with system prompt."""
    llm = get_llm()
    return (
        RunnablePassthrough() 
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) 
        | llm 
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items from transcript.
    
    Args:
        transcript: Meeting transcript text
        
    Returns:
        Formatted list of action items
        
    Raises:
        ValueError: If transcript is empty
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Extracting action items...")
    try:
        chain = build_chain(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all action items. For each provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, else write 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        )
        result = chain.invoke(transcript)
        logger.info("Action items extracted successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to extract action items: {str(e)}", exc_info=True)
        raise


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from transcript.
    
    Args:
        transcript: Meeting transcript text
        
    Returns:
        Formatted list of key decisions
        
    Raises:
        ValueError: If transcript is empty
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Extracting key decisions...")
    try:
        chain = build_chain(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all key decisions made. Format as a numbered list. "
            "If none found say 'No key decisions found.'"
        )
        result = chain.invoke(transcript)
        logger.info("Key decisions extracted successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to extract key decisions: {str(e)}", exc_info=True)
        raise


def extract_questions(transcript: str) -> str:
    """Extract open questions from transcript.
    
    Args:
        transcript: Meeting transcript text
        
    Returns:
        Formatted list of open questions
        
    Raises:
        ValueError: If transcript is empty
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    
    logger.info("Extracting open questions...")
    try:
        chain = build_chain(
            "From the meeting transcript, extract all unresolved questions "
            "or topics needing follow-up. Format as a numbered list. "
            "If none found say 'No open questions found.'"
        )
        result = chain.invoke(transcript)
        logger.info("Open questions extracted successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to extract questions: {str(e)}", exc_info=True)
        raise