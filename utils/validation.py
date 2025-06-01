import os
from typing import Optional
from pathlib import Path

def validate_safe_path(base_dir: str, file_path: str) -> Optional[str]:
    """
    Validate that a file path is safe and within the allowed directory.
    Returns normalized path if safe, None if unsafe.
    """
    try:
        base_path = Path(base_dir).resolve()
        file_path = Path(file_path).resolve()
        
        # Check if the path is within base directory
        if base_path in file_path.parents:
            return str(file_path)
        return None
    except Exception:
        return None

def validate_file_type(file_path: str) -> bool:
    """
    Validate file type and extension.
    """
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    try:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ALLOWED_EXTENSIONS
    except Exception:
        return False
