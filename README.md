# 🎬 AI Video Assistance

**Transform video content into actionable insights with AI-powered transcription, summarization, and RAG-based Q&A.**

## Overview

AI Video Assistance is an intelligent tool that processes YouTube videos or local media files and extracts:
- 📝 **Full Transcripts** — Multi-language support (English via Whisper, Hinglish via Sarvam)
- 🎯 **Smart Summaries** — Professional bullet-point summaries
- ❓ **Open Questions** — Unresolved questions highlighted
- 💬 **RAG Chat** — Ask questions about the video content with context-aware answers

## Features

✨ **Multi-Language Support**
- English: Powered by OpenAI Whisper (local, no API required)
- Hinglish: Powered by Sarvam AI (translates to English while transcribing)

🤖 **Modern AI Stack**
- LangChain for orchestration
- Mistral AI for LLM tasks
- Chroma vector database for RAG
- Sentence Transformers for embeddings

🎨 **Dual Interfaces**
- **Streamlit Web UI** — Professional dashboard with custom styling
- **CLI** — Perfect for automation and integration

## Prerequisites

Before you start, ensure you have:

### System Requirements
- **Python 3.13+**
- **FFmpeg** (required for audio processing)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### API Keys Required
1. **Mistral API Key** — [Get it here](https://console.mistral.ai/)
2. **Sarvam API Key** (optional, only for Hinglish) — [Get it here](https://www.sarvam.ai/)

## Installation

### 1. Clone/Setup the Project
```bash
cd "AI video assistance"
```

### 2. Create Virtual Environment
```bash
python3.13 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r Requirements.txt
# OR using uv (faster)
uv pip install -r Requirements.txt
```

### 4. Setup Environment Variables
Copy the template and add your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
nano .env  # or use your editor
```

**Required variables:**
```env
MISTRAL_API_KEY=your_mistral_key_here
WHISPER_MODEL=small  # Options: tiny, base, small, medium, large
SARVAM_API_KEY=your_sarvam_key_here  # Optional, for Hinglish
SARVAM_STT_MODEL=saaras:v2.5
```

## Usage

### Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Option 2: Command Line
```bash
python main.py
```
Follow the prompts to:
1. Enter a YouTube URL or local file path
2. Select language (english/hinglish)
3. Get instant results and chat with the transcript

### Example: Programmatic Usage
```python
from main import run_pipeline

result = run_pipeline(
    source="https://www.youtube.com/watch?v=example",
    language="english"
)

print(result["title"])
print(result["summary"])
print(result["action_items"])

# Chat with the transcript
rag_chain = result["rag_chain"]
answer = rag_chain.invoke("What are the next steps?")
print(answer)
```

## Project Structure

```
.
├── app.py                 # Streamlit web interface
├── main.py                # CLI entry point
├── Requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
│
├── core/
│   ├── transcriber.py     # Whisper & Sarvam transcription
│   ├── summarizer.py      # Meeting summarization
│   ├── extractor.py       # Extract actions, decisions, questions
│   ├── rag_engine.py      # RAG pipeline for Q&A
│   ├── vector_store.py    # Chroma vector database
│   └── text.txt           # Temporary storage
│
├── utils/
│   └── audio_process.py   # Audio/video processing & chunking
│
├── downloads/             # Downloaded video files
└── vector_db/             # Chroma vector database storage
    └── chroma.sqlite3
```

## Supported Input Formats

| Format | Source | Note |
|--------|--------|------|
| YouTube | URL | Requires internet access |
| MP3 | Local file | Auto-converted to WAV |
| WAV | Local file | Native format |
| MP4 | Local file | Audio extracted via FFmpeg |
| WebM | Local file | Auto-converted to WAV |

## Configuration

### Audio Chunking
Default: 10-minute chunks (in `utils/audio_process.py`)
```python
chunks = chunk_audio(wav_path, chunk_minutes=10)  # Adjust as needed
```

### Whisper Model Size
- `tiny` — Fastest, lower quality (~39M)
- `base` — Good balance (~140M)
- `small` — Better accuracy (~461M) ← **Default**
- `medium` — High accuracy (~1.5GB)
- `large` — Best accuracy (~2.9GB)

### RAG Retrieval
Default: 4 similar documents retrieved per query
Edit `core/rag_engine.py`:
```python
retriever = get_retriever(vector_store, k=4)  # Increase for more context
```

## Performance Tips

1. **Faster Transcription:** Use `tiny` or `base` Whisper model
2. **Faster Summarization:** Process smaller chunks or reduce `CHUNK_SIZE` in `core/summarizer.py`
3. **Lower Memory:** Reduce chunk overlap in vector store (`core/vector_store.py`)
4. **Parallel Processing:** Chunks are processed sequentially; consider async for production

## Troubleshooting

### ❌ "FFmpeg not found"
- **Solution:** Install FFmpeg (see Prerequisites)
- Verify: `ffmpeg -version`

### ❌ "MISTRAL_API_KEY not set"
- **Solution:** Ensure `.env` file exists with valid API key
- Check: `echo $MISTRAL_API_KEY`

### ❌ "Sarvam returned 401"
- **Solution:** Verify Sarvam API key in `.env`
- Note: Hinglish requires valid Sarvam subscription

### ❌ "Whisper model download fails"
- **Solution:** Check internet connection
- Manual download: `whisper --model small --model_dir ./models`

### ❌ "Out of Memory"
- **Solution:** Use smaller Whisper model (`tiny` or `base`)
- Reduce chunk size in `audio_process.py`

## Advanced: Docker Deployment

```dockerfile
FROM python:3.13
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY . .
RUN pip install -r Requirements.txt
CMD ["streamlit", "run", "app.py"]
```

Build & run:
```bash
docker build -t ai-video-assistance .
docker run -p 8501:8501 --env-file .env ai-video-assistance
```

## Limitations

- ⚠️ Requires stable internet for YouTube downloads
- ⚠️ Large files may timeout on API calls (>2GB)
- ⚠️ Sarvam API has 30-second audio limits per request
- ⚠️ Hinglish accuracy depends on pronunciation clarity

## Contributing

Found a bug? Have ideas? Open an issue or submit a PR!

## License

MIT License - feel free to use and modify.

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in terminal output
3. Verify all API keys are correctly set

---

**Happy analyzing! 🚀**
