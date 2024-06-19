# VSB: Vector Search Bench

<p align="center" width="100%">
   <img src=docs/images/splash.jpg width="180px"/>
</p>

**VSB** is a benchmarking suite for Vector Search. It lets you quickly measure how 
different workloads perform on a range of vector databases.

## Quickstart

### Requirements
* Python >= 3.11
* [Docker Compose](https://docs.docker.com/compose/) >= 2.27 (for non-cloud hosted databases)

> [!NOTE]
> macOS ships with an older version of Python (3.9 or earlier). Ensure you have
> a new enough version (e.g. via [Homebrew](https://brew.sh) - `brew install
> python@3.11`) otherwise VSB installation will fail.

### Install
1. Clone this repo:
   ```shell
   git clone https://github.com/pinecone-io/VSB.git
   ```
2. Use Poetry to install dependencies
   ```shell
   cd VSB
   pip3 install poetry && poetry install
   ``` 
3. Activate environment containing dependencies
   ```shell
   poetry shell
   ```

### Run

#### Cloud-hosted

To run VSB against a cloud-hosted vector database, simply
provide suitable credentials to an existing database instance. For example
to run workload _mnist-test_ against a Pinecone index:
```shell
vsb --database=pinecone --workload=mnist-test \
    --pinecone_api_key=<API_KEY> \
    --pinecone_index_name=<INDEX_NAME>
```
Where `--api_key` specifies the [Pinecone](https://app.pinecone.io) API key to use
and `--index_name` specifies the name of the index to connect to.

> [!TIP]
> The _mnist-test_ workload has dimensions=784 and metric=euclidean, if you
> don't have an existing index and need to create one via http://app.pinecone.io.

#### Local database via Docker

Alternatively, VSB can run against a locally-hosted vector database such as
pgvector running in Docker.

Launch the database via `docker compose` in one terminal:
```shell
cd docker/pgvector
docker compose up
```

From a second terminal run VSB:
```shell
vsb --database=pgvector --workload=mnist-test
```

Example output:
```shell
[2024-04-29 12:49:42,946] localhost/INFO/root: Using 10000 query vectors loaded from dataset 'queries' table
[2024-04-29 12:49:42,948] localhost/INFO/locust.runners: Ramping to 1 users at a rate of 10.00 per second
[2024-04-29 12:49:42,948] localhost/INFO/locust.runners: All users spawned: {"VectorSearchUser": 1} (1 total users)
[2024-04-29 12:49:44,314] localhost/INFO/root: Completed Load phase, switching to Run phase
[2024-04-29 12:49:45,687] localhost/INFO/root: User count: 1
[2024-04-29 12:49:45,687] localhost/INFO/root: Last user stopped, quitting runner
Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
Populate  mnist-test                                                                         3     0(0.00%) |    391     327     491    360 |    1.10        0.00
Search   mnist-test                                                                       100     0(0.00%) |     13      10      22     13 |   36.52        0.00
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                                       103     0(0.00%) |     24      10     491     13 |   37.61        0.00

Response time percentiles (approximated)
Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
Populate mnist-test                                                                            360    360    490    490    490    490    490    490    490    490    490      3
Search   mnist-test                                                                             13     13     14     14     15     16     19     22     22     22     22    100
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                                                                             13     13     14     14     16     18    330    360    490    490    490    103

```
