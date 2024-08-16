import configargparse
import argparse
import rich.table
import rich.console
import json
from pinecone import ServerlessSpec
from vsb.databases import Database
from vsb.workloads import Workload, WorkloadSequence
from vsb.vsb_types import DistanceMetric
from vsb import default_cache_dir


class WorkloadHelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "help":
            table = rich.table.Table(title="Available Workloads")
            table.add_column("Name", justify="left", no_wrap=True)
            table.add_column("Record Count", justify="right", style="green")
            table.add_column("Dimensions")
            table.add_column("Distance Metric", justify="center")
            table.add_column("Query Count", justify="right", style="red")
            for workload in Workload:
                if workload == Workload.Synthetic:
                    # Don't describe synthetic workload, static methods are not available
                    table.add_row("synthetic", "?", "?", "?", "?")
                    continue
                table.add_row(*tuple(str(x) for x in workload.describe()))
            console = rich.console.Console()
            console.print(table)
            parser.exit(0)
        else:
            setattr(namespace, self.dest, values)


def json_to_pinecone_spec(spec_string):
    try:
        spec = json.loads(spec_string)
        assert "pod" in spec or "serverless" in spec
        assert len(spec) == 1
        if "pod" in spec:
            assert "environment" in spec["pod"] and "pod_type" in spec["pod"]
        if "serverless" in spec:
            assert "cloud" in spec["serverless"] and "region" in spec["serverless"]
        return spec
    except Exception as e:
        raise ValueError from e


def add_vsb_cmdline_args(
    parser: configargparse.ArgumentParser, include_locust_args: bool
) -> None:
    """
    Add VSB's command-line arguments to `parser`.
    :param parser: Parser to add arguments to.
    :param include_locust_args: If True then also include existing locust
     arguments which VSB also supports.
    """
    main_group = parser.add_argument_group("Main arguments")
    main_group.add_argument(
        "--database",
        required=True,
        choices=tuple(e.value for e in Database),
        help="The vector search database to test",
    )
    main_group.add_argument(
        "--workload",
        action=WorkloadHelpAction,
        required=True,
        choices=tuple(e.value for e in Workload)
        + tuple(e.value for e in WorkloadSequence)
        + ("help",),
        help="The workload to run",
    )

    general_group = parser.add_argument_group("General options")
    general_group.add_argument(
        "--cache_dir",
        type=str,
        default=default_cache_dir,
        help="Directory to store downloaded datasets. Default is %(default)s).",
    )
    general_group.add_argument(
        "--log_dir",
        "-o",
        default="reports",
        help="Directory to write logs to. Default is %(default)s.",
    )
    general_group.add_argument(
        "--skip_populate",
        action="store_true",
        help="Skip the populate phase (useful if workload has already been loaded and is static)",
    )
    general_group.add_argument(
        "--requests_per_sec",
        type=float,
        default=0,
        help="Target requests per second for the Run phase. If using multiple users, "
        "then the target will be distributed across all users. "
        "Specify 0 for unlimited. Default is %(default)s.",
    )

    if include_locust_args:
        general_group.add_argument(
            "--loglevel",
            "-L",
            default="INFO",
            help="Choose between DEBUG/INFO/WARNING/ERROR/CRITICAL. Default is INFO",
            metavar="<level>",
        )

        general_group.add_argument(
            "--users",
            type=int,
            metavar="<int>",
            dest="num_users",
            default=1,
            help="Number of database clients to execute the workload. Default is %("
            "default)s",
        )
        general_group.add_argument(
            "--processes",
            type=int,
            help="Number of VSB subprocesses to fork and generate load from. Default "
            "is to run in a single process",
        )

    synthetic_group = parser.add_argument_group(
        "Options specific to synthetic workloads"
    )
    synthetic_group.add_argument(
        "--synthetic_record_count",
        type=int,
        default=10000,
        help="Number of records to generate for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_query_count",
        type=int,
        default=1000,
        help="Number of queries to generate for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_dimensions",
        type=int,
        default=192,
        help="Number of dimensions for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_metric",
        type=str,
        default="cosine",
        choices=tuple(e.value for e in DistanceMetric),
        help="Distance metric to use for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_top_k",
        type=int,
        default=10,
        help="Top-k value to use for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_seed",
        type=int,
        help="Seed to use for the synthetic workload. If not specified, a random seed will be generated.",
    )
    synthetic_group.add_argument(
        "--synthetic_steps",
        type=int,
        default=10,
        help="Number of steps to use for the synthetic workload. Default is %(default)s",
    )
    synthetic_group.add_argument(
        "--synthetic_no_aggregate_stats",
        action="store_true",
        help="Aggregate statistics for the synthetic workload. Default is %(default)s",
    )

    pinecone_group = parser.add_argument_group("Options specific to pinecone database")
    pinecone_group.add_argument(
        "--pinecone_api_key",
        type=str,
        help="API Key to connect to Pinecone index",
        env_var="VSB__PINECONE_API_KEY",
    )
    pinecone_group.add_argument(
        "--pinecone_index_name",
        type=str,
        default=None,
        help="Name of Pinecone index to connect to. One will be created if it does not exist. Default is vsb-<workload>.",
        env_var="VSB__PINECONE_INDEX_NAME",
    )

    pinecone_group.add_argument(
        "--pinecone_index_spec",
        type=json_to_pinecone_spec,
        default={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        help="JSON spec of Pinecone index to create (if it does not exist). Default is %(default)s",
    )

    pgvector_group = parser.add_argument_group("Options specific to pgvector database")
    pgvector_group.add_argument(
        "--pgvector_host",
        type=str,
        default="localhost",
        help="pgvector host to connect to. Default is %(default)s",
    )
    pgvector_group.add_argument(
        "--pgvector_port",
        type=str,
        default="5432",
        help="pgvector port to connect to. Default is %(default)s",
    )
    pgvector_group.add_argument(
        "--pgvector_database",
        type=str,
        help="pgvector database to use",
    )
    pgvector_group.add_argument(
        "--pgvector_username",
        type=str,
        default="postgres",
        help="Username to connect to pgvector index. Default is %(default)s",
        env_var="VSB__PGVECTOR_USERNAME",
    )
    pgvector_group.add_argument(
        "--pgvector_password",
        type=str,
        default="postgres",
        help="Password to connect to pgvector index. Default is %(default)s",
        env_var="VSB__PGVECTOR_PASSWORD",
    )
    pgvector_group.add_argument(
        "--pgvector_index_type",
        type=str,
        choices=["none", "ivfflat", "hnsw", "gin", "hnsw+gin", "ivfflat+gin"],
        default="hnsw",
        help="Index type to use for pgvector. Specifying 'none' will not create an "
        "ANN index, instead brute-force kNN search will be performed."
        "Default is %(default)s",
    )
    pgvector_group.add_argument(
        "--pgvector_ivfflat_lists",
        type=int,
        default=0,
        help="For pgvector IVFFLAT indexes the number of lists to create. A value of "
        "0 (default) means to automatically calculate based on the number of "
        "records R: R/1000 for up to 1M records, sqrt(R) for over 1M records.",
    )
    pgvector_group.add_argument(
        "--pgvector_search_candidates",
        type=int,
        default="0",  # 0 represents pgvector-recommended defaults (2*top_k for HNSW, sqrt(pgvector_ivfflat_lists) for IVFFLAT)
        help="Specify the size of the dynamic candidate list (ef_search for HNSW, probes for IVFFLAT). A higher value provides better recall at the cost of speed. Default is 2*top_k for HNSW and sqrt(pgvector_ivfflat_lists) for IVFFLAT",
    )
    pgvector_group.add_argument(
        "--pgvector_maintenance_work_mem",
        type=str,
        default="4GB",
        help=(
            "Set the postgres 'maintenance_work_mem' parameter - the amount of memory "
            "to use for maintenance operations such as CREATE INDEX. This should be "
            "at least as large as the index size. Specify as a string with size "
            "suffix (e.g. '2GB'). Default is %(default)s"
        ),
    )


def get_action(parser, argument_name):
    """Helper to lookup the named Action from the parser."""
    for action in parser._actions:
        if action.dest == argument_name:
            return action
    return None


def validate_parsed_args(
    parser: configargparse.ArgumentParser, args: configargparse.Namespace
):
    """Perform additional validation on parsed arguments, checking that any
    conditionally required arguments are present (e.g. --database=pinecone makes
    --pinecone_api_key required).
    If validation fails then parser.error() is called with an appropriate
    message, which will terminate the process.
    """

    match args.database:
        case "pinecone":
            required = (
                "pinecone_api_key",
                "pinecone_index_spec",
            )
            missing = list()
            for name in required:
                if not getattr(args, name):
                    missing.append(name)
            if missing:
                formatter = configargparse.HelpFormatter(".")
                formatter.start_section("")
                formatter.add_text("")
                for name in missing:
                    formatter.add_argument(get_action(parser, name))
                formatter.end_section()
                formatter.add_text(
                    "Please ensure all missing arguments are specified " "and re-run."
                )
                # Needed to ensure env var names are included in the actions'
                # help messages.
                parser.format_help()
                parser.error(
                    "The following arguments must be specified when --database is "
                    "'pinecone'" + formatter.format_help(),
                )
        case "pgvector":
            pass
        case _:
            pass
    match args.workload:
        case "synthetic":
            required = (
                "synthetic_record_count",
                "synthetic_query_count",
                "synthetic_dimensions",
                "synthetic_metric",
                "synthetic_top_k",
            )
            missing = list()
            for name in required:
                if not getattr(args, name):
                    missing.append(name)
            if missing:
                formatter = configargparse.HelpFormatter(".")
                formatter.start_section("")
                formatter.add_text("")
                for name in missing:
                    formatter.add_argument(get_action(parser, name))
                formatter.end_section()
                formatter.add_text(
                    "Please ensure all missing arguments are specified " "and re-run."
                )
                # Needed to ensure env var names are included in the actions'
                # help messages.
                parser.format_help()
                parser.error(
                    "The following arguments must be specified when --workload is "
                    "'synthetic'" + formatter.format_help(),
                )
        case _:
            pass
