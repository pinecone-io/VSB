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

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY> \
    --pinecone_index_name=<YOUR_INDEX_NAME> \
    --pinecone_index_spec=<YOUR_INDEX_SPEC>
```

> [!TIP]
> The API key and/or index name can also be passed via environment variables
> (`VSB__PINECONE_API_KEY` and `VSB__PINECONE_INDEX_NAME` respectively).
