from dotenv import load_dotenv
from utils.audio_process import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.logger_config import get_logger

logger = get_logger(__name__)

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    """Run the complete video processing pipeline.
    
    Args:
        source: YouTube URL or local file path
        language: Language for transcription ('english' or 'hinglish')
        
    Returns:
        Dictionary with title, transcript, summary, action items, decisions, questions, and RAG chain
        
    Raises:
        ValueError: If inputs are invalid
        Exception: If processing fails
    """
    # Input validation
    if not source or not source.strip():
        raise ValueError("Source cannot be empty")
    
    if language.lower() not in ["english", "hinglish"]:
        raise ValueError(f"Language must be 'english' or 'hinglish', got '{language}'")
    
    try:
        logger.info(f"Starting AI Video Assistant pipeline with source: {source}")
        
        chunks = process_input(source)
        logger.info(f"Processing {len(chunks)} audio chunk(s)...")

        transcript = transcribe_all(chunks, language)
        logger.info(f"Transcription complete. Length: {len(transcript)} characters")
        print(f"raw transcription (first 300 characters): {transcript[:300]}")

        title = generate_title(transcript)
        logger.info(f"Generated title: {title}")

        summary = summarize(transcript)
        logger.info("Generated summary")

        action_item = extract_action_items(transcript)
        logger.info("Extracted action items")

        decisions = extract_key_decisions(transcript)
        logger.info("Extracted key decisions")
        
        questions = extract_questions(transcript)
        logger.info("Extracted open questions")
        
        rag_chain = build_rag_chain(transcript)
        logger.info("Built RAG chain")

        logger.info("Pipeline completed successfully")
        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_item,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # CLI entry point
        source = input("Enter YouTube URL or local file path: ").strip()
        if not source:
            logger.error("Source cannot be empty")
            print("❌ Error: Source cannot be empty")
            exit(1)
        
        language = input("Language (english/hinglish): ").strip() or "english"
        if language.lower() not in ["english", "hinglish"]:
            logger.error(f"Invalid language: {language}")
            print(f"❌ Error: Language must be 'english' or 'hinglish', got '{language}'")
            exit(1)
        
        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print(f" Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        # Phase 2 — Chat with your meeting via RAG
        print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                logger.info("User ended chat session")
                print("👋 Goodbye!")
                break
            if not question:
                continue
            try:
                answer = ask_question(rag_chain, question)
                print(f"\n🤖 Assistant: {answer}\n")
            except Exception as e:
                logger.error(f"Error during RAG query: {str(e)}", exc_info=True)
                print(f"❌ Error: {str(e)}\n")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"❌ Fatal error: {str(e)}")
        exit(1)