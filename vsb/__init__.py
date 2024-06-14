import logging
from pathlib import Path

import rich.console
import rich.progress
import rich.live

logger = logging.getLogger("vsb")
"""Logger for VSB. Messages will be written to the log file and console."""


log_dir: Path = None
"""Directory where logs will be written to. Set in main()"""

console: rich.console.Console = None

progress: rich.progress.Progress = None
"""
Progress bar for the current task. Only created for non-Worker processes
(i.e. LocalRunner if non-distributed; Master if distributed).
"""

live: rich.live.Live = None
