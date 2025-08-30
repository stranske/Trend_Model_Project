import sys
import os
from pathlib import Path


def find_project_root(start_path: Path | None = None, markers: tuple[str, ...] | None = None) -> Path:
    """Find project root by searching for marker files up the directory tree.
    
    Args:
        start_path: Starting path for search (defaults to this file's directory)
        markers: File/directory names that indicate project root (defaults to common markers)
    
    Returns:
        Path to project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    if markers is None:
        markers = ("pyproject.toml", "requirements.txt", ".git", "setup.py", "setup.cfg")
    
    # Try environment variable override first
    env_root = os.environ.get("TREND_PROJECT_ROOT")
    if env_root:
        root_path = Path(env_root)
        if root_path.is_dir():
            return root_path
    
    # Search up the directory tree
    current = start_path
    for _ in range(10):  # Safety limit to prevent infinite loops
        for marker in markers:
            if (current / marker).exists():
                return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: if we can't find markers, assume we're in a typical structure
    # and try to find a reasonable root based on known paths
    fallback_path = start_path
    while fallback_path != fallback_path.parent:
        # Look for config directory as a fallback indicator
        if (fallback_path / "config").is_dir() and (fallback_path / "src").is_dir():
            return fallback_path
        fallback_path = fallback_path.parent
    
    raise FileNotFoundError(
        f"Cannot find project root from {start_path}. "
        f"Tried markers: {markers}. "
        "Set TREND_PROJECT_ROOT environment variable as override."
    )


# Robust path resolution using project root detection
try:
    ROOT = find_project_root()
except FileNotFoundError:
    # Fallback to original hardcoded path if root detection fails
    ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
