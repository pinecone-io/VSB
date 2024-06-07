#!/usr/bin/env python3
import sys

import locust.stats
import vsb.metrics_tracker

# Monkey-patch locust's print_stats_json function to include our additional metrics.
locust.stats.print_stats_json = vsb.metrics_tracker.print_stats_json

import configargparse
import locust.main

from vsb.cmdline_args import add_vsb_cmdline_args, validate_parsed_args


def main():
    # We use locust internally to drive execution, however we want to present
    # our own benchmark-centric command-line arguments to the user; not the
    # large number of highly configurable arguments locust provides.
    #
    # As such, we define our own argument parser and parse the user's options
    # before locust - this ensures that 'vsb.py --help' only shows the relevant
    # benchmarking arguments. To ensure we can later consume those arguments
    # inside VSB, we _also_ need to add the arguments to locust's own
    # parser, which is done by adding a listener to init_command_line_parser
    # inside vsb/locustfile.py which calls the same add_cmdline_args() method
    # as below.
    parser = configargparse.ArgumentParser(
        prog="vsb.py",
        description="Vector Search Bench",
        usage="vsb.py --database=<DATABASE> --workload=<WORKLOAD> [additional "
        "options...]\nPass --help for full list of options.\n",
    )
    add_vsb_cmdline_args(parser, include_locust_args=True)

    # Parse options and validate arguments passed, and to print the vsb usage
    # message (and exit) if args fail validation or --help passed.
    args = parser.parse_args()
    validate_parsed_args(parser, args)

    # If we got here then args are valid - pass them on to locusts' main(),
    # appending the location of our locustfile and --headless to start
    # running immediately.
    sys.argv += ["-f", "./vsb/locustfile.py", "--headless"]
    locust.main.main()


if __name__ == "__main__":
    main()
