#!/usr/bin/env python3

import sys
from pathlib import Path

import configargparse
import locust.main

import vsb
from vsb.cmdline_args import add_vsb_cmdline_args, validate_parsed_args
from vsb.logging import logger, setup_logging


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
        conflict_handler="resolve",
    )
    add_vsb_cmdline_args(parser, include_locust_args=True)

    # Parse options and validate arguments passed, and to print the vsb usage
    # message (and exit) if args fail validation or --help passed.
    args = parser.parse_args()
    validate_parsed_args(parser, args)

    log_base = Path(args.log_dir) / args.database
    vsb.log_dir = setup_logging(log_base=log_base, level=args.loglevel)
    requests_per_sec = (
        "{:g}".format(args.requests_per_sec) if args.requests_per_sec else "unlimited"
    )
    logger.info(
        f"Vector Search Bench: Starting experiment with backend='{args.database}', "
        f"workload='{args.workload}', users={args.num_users}, requests_per_sec={requests_per_sec}"
    )
    logger.info(f"Writing benchmark results to '{vsb.log_dir}'")
    if args.workload == "synthetic-proportional":
        logger.warning(
            "SyntheticProportionalWorkloads don't have ground-truth based metrics like recall yet."
        )

    # If we got here then args are valid - pass them on to locusts' main(),
    # appending the location of our locustfile and --headless to start
    # running immediately.
    sys.argv += ["-f", "./vsb/locustfile.py", "--headless", "--skip-log-setup"]
    locust.main.main()


if __name__ == "__main__":
    main()
