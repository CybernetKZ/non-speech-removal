# Audio Speech Trimming Service

A FastAPI service that uses Silero VAD (Voice Activity Detection) to detect and extract speech segments from audio files, removing non-speech parts while maintaining natural transitions.

## Features

- **High-accuracy speech detection** using Silero VAD
- **Flexible device support** - runs on CPU or GPU
- **Natural audio transitions** with configurable silence gaps
- **Multiple audio format support** via pydub
- **RESTful API** with FastAPI

## Device Configuration

You can control whether the service runs on CPU or GPU by setting the `DEVICE` environment variable:

### CPU Usage (Default)
```bash
export DEVICE=cpu
python src/main.py
```

### GPU Usage
```bash

export DEVICE=cuda


export DEVICE=cuda:0
export DEVICE=cuda:1

python src/main.py
```

### Usage
```shell
docker compose up -d
```

The service will start on `http://localhost:6070`

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check including device info

### Process Audio
- `POST /process-audio` - Upload and process audio file

Parameters:
- `file`: Audio file (required)
- `speech_threshold`: Speech detection threshold (0.0-1.0, default: 0.5)

## Example Usage

```python
import requests

# Upload and process audio file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process-audio',
        files={'file': f},
        params={'speech_threshold': 0.5}
    )

# Save processed audio
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

## Configuration Options

- `DEVICE`: Device for inference ("cpu", "cuda", "cuda:0", etc.)
- `speech_threshold`: Sensitivity of speech detection (lower = more sensitive)

## Requirements

- Python 3.10+
- PyTorch (with CUDA support for GPU usage)
- See `requirements.txt` for full list

## Performance Notes

- **GPU acceleration**: Significantly faster processing for larger audio files
- **CPU mode**: More compatible, lower memory usage
- **Memory usage**: GPU mode requires additional VRAM
- **First run**: Model download may take time on first startup 