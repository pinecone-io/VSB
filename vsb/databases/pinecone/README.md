# Pinecone

This directory adds support running experiments against
[Pinecone](https://www.pinecone.io) - a managed vector database.

It supports connecting to both Pod-based and Serverless indexes.

To run VSB against a Pinecone index:

1. Invoke VSB with `--database=pinecone` and provide your API key to VSB.
   A serverless index will be created for you, using aws/us-east-1.

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY>
```

If you wish to configure the created index or use an already existing one,
specify the name and/or spec with `--pinecone_index_name` and `--pinecone_index_spec`.

The `--pinecone_index_spec` option takes a JSON string described in the [Pinecone docs](https://docs.pinecone.io/reference/api/control-plane/create_index).

You can specify a [Namespace](https://docs.pinecone.io/guides/index-data/indexing-overview#namespaces) using `--pinecone_namespace_name`. If no namespace is provided, the data will be loaded into the `__default__` namespace.

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY> \
    --pinecone_index_name=<YOUR_INDEX_NAME> \
    --pinecone_index_spec=<YOUR_INDEX_SPEC> \
    --pinecone_namespace_name=<YOUR_NAMESPACE_NAME>
```

## Dedicated Read Nodes

VSB supports creating Pinecone serverless indexes with [dedicated read nodes](https://docs.pinecone.io/guides/indexes/dedicated-read-nodes) for improved read performance and isolation.

To create an index with dedicated read nodes, use the `--pinecone_dedicated_read_nodes` flag along with optional configuration parameters:

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY> \
    --pinecone_dedicated_read_nodes \
    --pinecone_dedicated_node_type=b1 \
    --pinecone_dedicated_shards=2 \
    --pinecone_dedicated_replicas=1
```

Available dedicated read node options:
- `--pinecone_dedicated_read_nodes`: Enable dedicated read nodes (default: False)
- `--pinecone_dedicated_node_type`: Node type (e.g., b1, b2). Default is b1
- `--pinecone_dedicated_shards`: Number of shards. Default is 1
- `--pinecone_dedicated_replicas`: Number of replicas. Default is 1

> [!NOTE]
> Dedicated read nodes are only available for serverless indexes and require Pinecone API version 2025-10 or later.

> [!TIP]
> The API key and/or index name can also be passed via environment variables
> (`VSB__PINECONE_API_KEY` and `VSB__PINECONE_INDEX_NAME` respectively).

## Multi-Namespace Benchmarking

VSB supports benchmarking across multiple namespaces in an existing Pinecone index. When `--pinecone_multi_namespace=True`, VSB automatically discovers all populated namespaces in the index and distributes requests evenly across them.

**Requirements:**
- The index must already exist and be populated
- Must use `--skip_populate` (multi-namespace mode only benchmarks existing indexes)
- Must specify `--pinecone_index_name` (cannot auto-generate index name)
- Cannot use `--overwrite` or custom `--pinecone_namespace_name` with multi-namespace mode

**Usage:**

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY> \
    --pinecone_index_name=<EXISTING_INDEX> \
    --pinecone_multi_namespace=True \
    --skip_populate
```

**How it works:**
- VSB automatically discovers all namespaces in the index that have records (`record_count > 0`)
- Namespaces are distributed across users/workers using a round-robin algorithm
- Requests from each user are distributed evenly across their assigned namespaces
- Metrics are aggregated across all namespaces (same display format as single-namespace mode)

**Example:** If an index has 3 namespaces (ns1, ns2, ns3) and you run with 2 users:
- User 0 will handle requests to ns1 and ns3
- User 1 will handle requests to ns2
- Requests are distributed evenly across the assigned namespaces for each user
