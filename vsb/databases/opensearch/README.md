# Amazon OpenSearch

This directory adds support running experiments against Opensearch or 
[Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) - a managed vector database.

It supports connecting to both Serverless collections and Managed cluster collections. Use `--opensearch_service` to specify the service type, 'aoss' for Amazon OpenSearch Serverless and 'es' for Amazon OpenSearch Managed cluster.

To run VSB against a local Opensearch instance:

1. Start a local Opensearch instance using Docker:
```shell
cd docker/opensearch
docker-compose up -d
```
2. Invoke VSB with `--database=opensearch` and the default username and port to VSB. 
   Note for the local Docker container, TLS is disabled:
```shell
vsb --database=opensearch --workload=mnist-test \
    --no-opensearch_use_tls \ 
    --opensearch_username=admin \
    --opensearch_password=opensearch \
```

To run VSB against a Amazon OpenSearch collections:

1. Invoke VSB with `--database=opensearch` and provide your AWS Access Credentials to VSB.
   A serverless index will be created for you in the collection.
   Please note: [Amazon OpenSearch collection](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-getting-started.html) should be created before getting started with VSB

```shell
vsb --database=opensearch --workload=mnist-test \
    --opensearch_host=<YOUR_OPENSEARCH_HOST> \
    --opensearch_region=<YOUR_OPENSEARCH_REGION> \
    --opensearch_service=<YOUR_OPENSEARCH_SERVICE> \
    --aws_access_key=<YOUR_AWS_ACCESS_KEY> \
    --aws_secret_key=<YOUR_AWS_SECRET_KEY> \
    --aws_session_token=<YOUR_AWS_SESSION_TOKEN>
```
