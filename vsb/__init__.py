import logging
from pathlib import Path

logger = logging.getLogger("vsb")

log_dir: Path = None
"""Directory where logs will be written to. Set in main()"""
