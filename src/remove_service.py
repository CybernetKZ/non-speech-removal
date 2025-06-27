import logging
from pydub import AudioSegment
import io
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


def detect_speech_segments(audio_stream, vad_model, speech_threshold=0.5, device="cpu"):
    """
    Detect speech segments in audio using Silero VAD.
    Returns a list of (start, end) timestamps in seconds for speech segments.
    
    Args:
        audio_stream: Audio file stream
        vad_model: Loaded Silero VAD model
        speech_threshold: Threshold for speech detection (0.0-1.0)
        device: Device to run inference on ("cpu", "cuda", etc.)
    """
    
    _, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    try:
        
        audio_stream.seek(0)
        
        
        audio = AudioSegment.from_file(audio_stream)
        
        
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        
        if audio.sample_width == 2:  
            samples = samples / 32768.0
        elif audio.sample_width == 4:  
            samples = samples / 2147483648.0
        
        
        wav_tensor = torch.from_numpy(samples).to(device)
        
        
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            vad_model,
            threshold=speech_threshold,
            sampling_rate=16000,
            return_seconds=True
        )
        
        
        speech_segments = []
        for segment in speech_timestamps:
            start = segment['start']
            end = segment['end']
            speech_segments.append((start, end))
        
        logger.info(f"Detected {len(speech_segments)} speech segments using {device}")
        return speech_segments
        
    except Exception as e:
        logger.error(f"Error in speech detection on {device}: {str(e)}")
        return []


def remove_non_speech_segments(audio_stream, speech_segments, silence_gap=0.15):
    """
    Remove non-speech segments from audio and replace with 0.15s silence gaps only where segments were removed.
    """
    audio_stream.seek(0)
    audio = AudioSegment.from_file(audio_stream)
    
    if not speech_segments:
        return None
    
    
    speech_segments = sorted(speech_segments, key=lambda x: x[0])
    
    
    result_audio = AudioSegment.empty()
    silence = AudioSegment.silent(duration=int(silence_gap * 1000))
    
    for i, (start, end) in enumerate(speech_segments):
        
        segment = audio[int(start * 1000):int(end * 1000)]
        result_audio += segment
        
        
        if i < len(speech_segments) - 1:
            next_start = speech_segments[i + 1][0]
            
            if next_start > end:
                result_audio += silence
    
    
    output_stream = io.BytesIO()
    result_audio.export(output_stream, format="wav")
    output_stream.seek(0)
    
    return output_stream
