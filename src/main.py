import os
import tempfile
import logging
import uuid
from contextlib import asynccontextmanager
import time
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

import torch
from utils import clear_memory
from remove_service import detect_speech_segments, remove_non_speech_segments


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
)
logger = logging.getLogger(__name__)


vad_model = None
temp_dir = None

DEVICE = os.getenv("DEVICE", "cuda")  # Options: "cpu", "cuda", "cuda:0", etc.

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad_model, temp_dir
    
    logger.info("Starting up Audio Processing Service...")
    
    temp_dir = tempfile.mkdtemp(prefix="audio_service_")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        logger.info(f"Loading Silero VAD model on device: {DEVICE}")
        torch.set_num_threads(1)
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        vad_model = vad_model.to(DEVICE)
        logger.info(f"Silero VAD model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load Silero VAD model: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down Audio Processing Service...")
    
    if vad_model:
        del vad_model
        clear_memory()
    
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

app = FastAPI(
    title="Audio Speech Trimming Service",
    description="Service to detect and extract speech segments from audio files with natural transitions using Silero VAD",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Audio Speech Trimming Service",
        "status": "running",
        "model": "Silero VAD",
        "device": DEVICE
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global vad_model
    return {
        "status": "healthy",
        "model_loaded": vad_model is not None,
        "temp_dir": temp_dir,
        "model_type": "Silero VAD",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/process-audio")
async def process_audio_download(
    file: UploadFile = File(...),
    speech_threshold: float = 0.5
):
    """
    Process an audio file to remove non-speech segments and return only speech parts.
    Uses Silero VAD for speech detection.
    """
    import mimetypes
    start_time = time.time()

    
    if file.content_type and not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        logger.info(f"Processing file: {file.filename}")
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        
        speech_segments = detect_speech_segments(
            audio_stream, 
            vad_model,
            speech_threshold=speech_threshold,
            device=DEVICE
        )
        
        if not speech_segments:
            return Response(content="No speech segments detected", status_code=200)

        
        output_stream = remove_non_speech_segments(
            audio_stream, 
            speech_segments
        )

        if not output_stream:
            raise HTTPException(status_code=500, detail="Failed to process audio")

        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {file.filename} in {processing_time:.2f}s")

        
        base_name = os.path.splitext(file.filename or "audio")[0]
        out_filename = f"speech_only_{base_name}.wav"
        mime_type = mimetypes.guess_type(out_filename)[0] or 'audio/wav'

        return StreamingResponse(
            output_stream,
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={out_filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        clear_memory()
        

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
