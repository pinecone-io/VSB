"""
Logging support for VSB.
Includes setup code for logging and rich console output for progress bars etc.

Log messages from VSB should use the module-level `vsb.logger` for the actual
logger object - e.g.

    from vsb import logger

    logger.info("This is an info message")
"""

import io
import logging
import os
from datetime import datetime
from pathlib import Path
import rich.console
import rich.live
from rich.logging import RichHandler
from rich.progress import Progress
import rich.table
import vsb

logger = logging.getLogger("vsb")

progress_greenlet = None


class ExtraInfoColumn(rich.progress.ProgressColumn):
    """A custom rich.progress column which renders an extra_info field
    of a task if the field is present, otherwise shows nothing.
    extra_info field can include rich markup (e.g. [progress.description] etc).
    """

    def render(self, task: rich.progress.Task) -> rich.text.Text:
        # Check if the task has the extra_info field
        if "extra_info" in task.fields:
            return rich.text.Text.from_markup(task.fields["extra_info"])
        return rich.text.Text()


class ProgressIOWrapper(io.IOBase):
    """A wrapper around a file-like object which updates a progress bar as data is
    written to the file.
    """

    def __init__(self, dest, total, progress, indent=0, *args, **kwargs):
        """Create a new ProgressIOWrapper object.
        :param dest: The destination file-like object to write to.
        :param total: The total number of bytes expected to be written (used to
                      percentage complete). Pass None if unknown - this won't show
                      a percentage complete, but will otherwise track progress.
        :param progress: The Progress object to add a progress bar to. If None
                         then no progress bar will be shown.
        :param indent: The number of spaces to indent the progress bar label
        """
        self.path = dest
        self.file = dest.open("wb")
        self.progress = progress
        if self.progress:
            self.task_id = progress.add_task(
                (" " * indent) + dest.parent.name + "/" + dest.name, total=total
            )
        super().__init__(*args, **kwargs)

    def write(self, b):
        # Write data to the base object
        bytes_written = self.file.write(b)
        if self.progress:
            # Update the progress bar with the amount written
            self.progress.update(self.task_id, advance=bytes_written)
        return bytes_written

    def flush(self):
        return self.file.flush()

    def close(self):
        return self.file.close()

    # Implement other necessary methods to fully comply with IO[bytes] interface
    def seek(self, offset, whence=io.SEEK_SET):
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def read(self, n=-1):
        return self.file.read(n)

    def readable(self):
        return self.file.readable()

    def writable(self):
        return self.file.writable()

    def seekable(self):
        return self.file.seekable()


def setup_logging(log_base: Path, level: str) -> Path:
    level = level.upper()
    # Setup the default logger to log to a file under
    # <log_base>/<timestamp>/vsb.log,
    # returning the directory created.
    log_path = log_base / datetime.now().isoformat(timespec="seconds")
    log_path.mkdir(parents=True)

    file_handler = logging.FileHandler(log_path / "vsb.log")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_handler.setFormatter(file_formatter)

    # Configure the root logger to use the file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    # Setup a rich text console for log messages and progress bars etc.
    # Set a fixed width of 300 to disable wrapping if not in a terminal (for CV
    # so log lines we are checking for don't get wrapped).
    width = None if os.getenv("TERM") else 300
    vsb.console = rich.console.Console(width=width)
    # Setup the specific logger for "vsb" to also log to stdout using RichHandler
    rich_handler = RichHandler(
        console=vsb.console,
        log_time_format="%Y-%m-%dT%H:%M:%S%z",
        omit_repeated_times=False,
        show_path=False,
    )
    rich_handler.setLevel(level)
    vsb.logger.setLevel(level)
    vsb.logger.addHandler(rich_handler)

    # And always logs errors to stdout (via RichHandler)
    error_handler = RichHandler()
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    return log_path


def make_progressbar() -> rich.progress.Progress:
    """Create a Progress object for use in displaying progress bars.
    To display the progress of one or more tasks, call add_task() on the returned
    object, then call update() or advance() to advance progress -

        progress = make_progressbar()
        task_id = progress.add_task("Task description", total=100)
        progress.update(task_id, advance=1)
    """
    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.MofNCompleteColumn(),
        rich.progress.BarColumn(),
        rich.progress.TaskProgressColumn(),
        "[progress.elapsed]elapsed:",
        rich.progress.TimeElapsedColumn(),
        "[progress.remaining]remaining:",
        rich.progress.TimeRemainingColumn(compact=True),
        ExtraInfoColumn(),
        console=vsb.console,
    )
    progress.start()
    return progress
