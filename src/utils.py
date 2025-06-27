import os
import gc
import logging

logger = logging.getLogger(__name__)

def clear_memory():
    """Clear CUDA cache and run garbage collection."""

    gc.collect()

def cleanup_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")
