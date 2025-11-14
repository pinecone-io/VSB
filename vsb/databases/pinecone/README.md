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
