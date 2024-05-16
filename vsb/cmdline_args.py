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
    parser.add_argument(
        "--database",
        required=True,
        choices=tuple(e.value for e in Database),
        help="The vector search database to test",
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=tuple(e.value for e in Workload),
        help="The workload to run",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/VSB/cache",
        help="Directory to store downloaded datasets",
    )
    parser.add_argument(
        "--skip_populate",
        action="store_true",
        help="Skip the populate phase (useful if workload has already been loaded and is static)",
    )

    if include_locust_args:
        parser.add_argument(
            "--json",
            default=False,
            action="store_true",
            help="Prints the final stats in JSON format to stdout. Useful for parsing "
            "the results in other programs/scripts. Use together with --headless "
            "and --skip-log for an output only with the json data.",
        )

        parser.add_argument(
            "--loglevel",
            "-L",
            default="INFO",
            help="Choose between DEBUG/INFO/WARNING/ERROR/CRITICAL. Default is INFO.",
            metavar="<level>",
        )

        parser.add_argument(
            "--users",
            type=int,
            default=1,
            help="Number of database clients to execute the workload",
        )
        parser.add_argument(
            "--processes",
            type=int,
            help="Number of VSB subprocesses to fork and generate load from",
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
