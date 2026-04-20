#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import configargparse
import locust.main

import vsb
from vsb.cmdline_args import add_vsb_cmdline_args, validate_parsed_args
from vsb.logging import logger, setup_logging

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


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

    # Auto-detect synthetic dimensions from an existing Pinecone index.
    if (
        args.database == "pinecone"
        and args.workload.startswith("synthetic")
        and getattr(args, "pinecone_index_name", None) is not None
        and "--synthetic_dimensions" not in sys.argv
    ):
        try:
            from pinecone.grpc import PineconeGRPC
            pc = PineconeGRPC(args.pinecone_api_key)
            index_info = pc.describe_index(args.pinecone_index_name)
            index_dims = index_info["dimension"]
            args.synthetic_dimensions = index_dims
            sys.argv += ["--synthetic_dimensions", str(index_dims)]
            logger.info(
                f"Auto-detected synthetic_dimensions={index_dims} "
                f"from index '{args.pinecone_index_name}'"
            )
        except Exception as e:
            logger.warning(
                f"Could not auto-detect dimensions from index "
                f"'{args.pinecone_index_name}': {e}"
            )

    # Auto-calculate the number of users if not explicitly specified.
    # Assuming a conservative 500ms request latency, each user can issue
    # at most 2 requests/sec. We provision enough users to comfortably
    # achieve the target QPS.
    if args.num_users is None:
        if args.requests_per_sec > 0:
            assumed_latency = 0.5  # 500ms
            args.num_users = max(1, math.ceil(args.requests_per_sec * assumed_latency))
        else:
            args.num_users = 1
        sys.argv += ["--users", str(args.num_users)]

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
