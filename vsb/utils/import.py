from pinecone import Pinecone, ImportErrorMode
from dotenv import load_dotenv
import argparse
import os

load_dotenv()


def main():
    """
    This script is used to import data into a Pinecone index utilizing the bulk import feature.
    It is useful for large datasets that do not fit in memory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-name",
        type=str,
        required=False,
        default=os.getenv("PINECONE_INDEX_NAME"),
        help="Name of the index to import to. If not provided, it will be inferred from the environment variable PINECONE_INDEX_NAME.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root of the bucket to import from. Provide either s3://<bucket-name>/<path-to-data> or <bucket-name>/<path-to-data>.",
    )
    parser.add_argument(
        "--integration-id",
        type=str,
        required=True,
        help="Integration ID for the bucket. This can be found in the Pinecone console.",
    )
    args = parser.parse_args()

    pc = Pinecone()
    index_name = args.index_name or f"vsb-{os.getenv('VSB_WORKLOAD')}"
    if args.root.startswith("s3://"):
        root = args.root
    else:
        root = f"s3://{args.root}"
    if os.getenv("PINECONE_HOST") is None:
        index = pc.Index(name=index_name)
    else:
        index = pc.Index(host=os.getenv("PINECONE_HOST"), name=index_name)

    index.start_import(
        uri=root,
        error_mode=ImportErrorMode.CONTINUE,  # or ImportErrorMode.ABORT
        integration_id=args.integration_id,  # Optional for public buckets
    )


if __name__ == "__main__":
    main()
