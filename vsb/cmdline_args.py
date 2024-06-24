import configargparse
from vsb.databases import Database
from vsb.workloads import Workload


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
        required=True,
        choices=tuple(e.value for e in Workload),
        help="The workload to run",
    )

    general_group = parser.add_argument_group("General options")
    general_group.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/VSB/cache",
        help="Directory to store downloaded datasets. Default is %(default)s).",
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
        help="Name of Pinecone index to connect to",
        env_var="VSB__PINECONE_INDEX_NAME",
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
        choices=["ivfflat", "hnsw"],
        default="hnsw",
        help="Index type to use for pgvector. Default is %(default)s",
    )
    pgvector_group.add_argument(
        "--pgvector_ivfflat_lists",
        type=int,
        default="100",
        help="For pgvector IVFFLAT indexes, number of lists to create. Default is %(default)s",
    )
    pgvector_group.add_argument(
        "--pgvector_search_candidates",
        type=int,
        default="0",  # 0 represents pgvector-recommended defaults (2*top_k for HNSW, sqrt(pgvector_ivfflat_lists) for IVFFLAT)
        help="Specify the size of the dynamic candidate list (ef_search for HNSW, probes for IVFFLAT). A higher value provides better recall at the cost of speed. Default is 2*top_k for HNSW and sqrt(pgvector_ivfflat_lists) for IVFFLAT",
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
            required = ("pinecone_api_key", "pinecone_index_name")
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
