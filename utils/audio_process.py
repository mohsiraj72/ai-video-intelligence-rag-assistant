import yt_dlp
from pydub import AudioSegment
import os
from core.logger_config import get_logger

logger = get_logger(__name__)

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube URL and convert to WAV format.
    
    Args:
        url: YouTube URL
        
    Returns:
        Path to downloaded WAV file
        
    Raises:
        ValueError: If URL is empty or invalid
        Exception: If download fails
    """
    if not url or not url.strip():
        raise ValueError("YouTube URL cannot be empty")
    
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL format: {url}")
    
    logger.info(f"Downloading audio from: {url}")
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
        logger.info(f"Successfully downloaded: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download YouTube audio: {str(e)}")
        raise



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub.
    
    Args:
        input_path: Path to input audio/video file
        
    Returns:
        Path to converted WAV file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: If conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    logger.info(f"Converting {input_path} to WAV...")
    try:
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz mono
        audio.export(output_path, format="wav")
        logger.info(f"Successfully converted to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert file: {str(e)}")
        raise



def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split audio file into chunks.
    
    Args:
        wav_path: Path to WAV file
        chunk_minutes: Duration of each chunk in minutes (default: 10)
        
    Returns:
        List of chunk file paths
        
    Raises:
        FileNotFoundError: If WAV file doesn't exist
        ValueError: If chunk_minutes is invalid
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")
    
    if chunk_minutes <= 0:
        raise ValueError(f"chunk_minutes must be positive, got {chunk_minutes}")
    
    logger.info(f"Chunking audio into {chunk_minutes}-minute segments...")
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    logger.info(f"Created {len(chunks)} chunk(s)")
    return chunks

def process_input(source: str) -> list:
    """Process input source (YouTube URL or local file) into audio chunks.
    
    Args:
        source: YouTube URL or local file path
        
    Returns:
        List of audio chunk file paths
        
    Raises:
        ValueError: If source is empty or invalid
        FileNotFoundError: If local file doesn't exist
        Exception: If processing fails
    """
    if not source or not source.strip():
        raise ValueError("Source cannot be empty")
    
    try:
        if source.startswith("http://") or source.startswith("https://"):
            logger.info("Detected YouTube URL. Downloading audio...")
            wav_path = download_youtube_audio(source)
        else:
            logger.info("Detected local file. Converting to WAV...")
            wav_path = convert_to_wav(source)

        logger.info("Chunking audio...")
        chunks = chunk_audio(wav_path)
        logger.info(f"Audio ready — {len(chunks)} chunk(s) created.")
        return chunks
    except Exception as e:
        logger.error(f"Failed to process input: {str(e)}")
        raise