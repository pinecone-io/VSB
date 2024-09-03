"""
Sometimes during tests, the Locust process hangs/times out and is unable to clean up the index it creates in Pinecone.
This can interfere with subsequent tests by leaving pod indices in the DaveR-VSB-CI project, not to mention keeping
the project cluttered with unnecessary indices. This script is intended to be run after each test to clean up any
remaining indices in the project. 
"""

from pinecone import Pinecone
import os

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

is_pod_run = os.getenv("ENVIRONMENT") == "pod"
prefix = os.getenv("NAME_PREFIX")

for index in pc.list_indexes():
    if (
        index["name"][: len(prefix)] == prefix
        and (is_pod_run and "pod" in index["spec"])
        or (not is_pod_run and "serverless" in index["spec"])
    ):
        print(f"Deleting index {index}")
        pc.delete_index(index)
