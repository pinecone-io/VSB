# Pinecone

This directory adds support running experiments against
[Pinecone](https://www.pinecone.io) - a managed vector database.

It supports connecting to both Pod-based and Serverless indexes.

To run VSB against a Pinecone index:

1. Create an index via the [Pinecone console](https://app.pinecone.io) with the
   appropriate dimensionality & metric - e.g. for `mnist-test` use `dimensions=784` and
   `metric=euclidean`.
2. Invoke VSB with `--database=pinecone` and provide the API key and index name to VSB.

```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<YOUR_API_KEY> \
    --pinecone_index_name=<YOUR_INDEX_NAME>
```

> [!TIP]
> The API key and/or index name can also be passed via environment variables
> (`VSB__PINECONE_API_KEY` and `VSB__PINECONE_INDEX_NAME` respectively).
