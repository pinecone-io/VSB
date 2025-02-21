# Amazon OpenSearch

This directory adds support running experiments against
[Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) - a managed vector database.

It currently only supports connecting to Serverless collections. Managed cluster collections would require a nodes provisioing, cluster set up and different implementation, which would be supported in future.

To run VSB against a Amazon OpenSearch collections:

1. Invoke VSB with `--database=opensearch` and provide your AWS Access Credentials to VSB.
   A serverless index will be created for you in the collection.
   Please note: [Amazon OpenSearch collection](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-getting-started.html) should be created before getting started with VSB

```shell
vsb --database=opensearch --workload=mnist-test \
    --opensearch_host=<YOUR_OPENSEARCH_HOST> \
    --opensearch_region=<YOUR_OPENSEARCH_REGION> \
    --aws_access_key=<YOUR_AWS_ACCESS_KEY> \
    --aws_secret_key=<YOUR_AWS_SECRET_KEY> \
    --aws_session_token=<YOUR_AWS_SESSION_TOKEN>
```
