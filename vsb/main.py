#!/usr/bin/env python3
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import configargparse
import locust.main
from rich.console import Console
from rich.logging import RichHandler

import vsb.metrics_tracker
from vsb.cmdline_args import add_vsb_cmdline_args, validate_parsed_args


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

    # Setup the specific logger for "vsb" to also log to stdout using RichHandler
    # set fixed width of 300 to disable wrapping if not in a terminal (for CV
    # so log lines we are checking for don't get wrapped).
    width = None if os.getenv("TERM") else 300
    rich_handler = RichHandler(
        console=Console(width=width),
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


def main():
    # We use locust internally to drive execution, however we want to present
    # our own benchmark-centric command-line arguments to the user; not the
    # large number of highly configurable arguments locust provides.
    #
    # As such, we define our own argument parser and parse the user's options
    # before locust - this ensures that 'vsb --help' only shows the relevant
    # benchmarking arguments. To ensure we can later consume those arguments
    # inside VSB, we _also_ need to add the arguments to locust's own
    # parser, which is done by adding a listener to init_command_line_parser
    # inside vsb/locustfile.py which calls the same add_cmdline_args() method
    # as below.
    parser = configargparse.ArgumentParser(
        prog="vsb",
        description="Vector Search Bench",
        usage="vsb --database=<DATABASE> --workload=<WORKLOAD> [additional "
        "options...]\nPass --help for full list of options.\n",
    )
    add_vsb_cmdline_args(parser, include_locust_args=True)

    # Parse options and validate arguments passed, and to print the vsb usage
    # message (and exit) if args fail validation or --help passed.
    args = parser.parse_args()
    validate_parsed_args(parser, args)

    log_base = Path("reports") / args.database
    vsb.log_dir = setup_logging(log_base=log_base, level=args.loglevel)
    vsb.logger.info(f"Writing benchmark results to '{vsb.log_dir}'")

    # If we got here then args are valid - pass them on to locusts' main(),
    # appending the location of our locustfile and --headless to start
    # running immediately.
    sys.argv += ["-f", "./vsb/locustfile.py", "--headless", "--skip-log-setup"]
    locust.main.main()


if __name__ == "__main__":
    main()
