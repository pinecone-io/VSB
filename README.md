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
    --pinecone_api_key=<API_KEY> 
```
Where `--api_key` specifies the [Pinecone](https://app.pinecone.io) API key to use.

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

<p align="center" width="100%">
   <img src=docs/images/vsb_example_output.png/>
</p>

## Overview

VSB` is a cli tool used to run various workloads against a range of 
vector databases, and measure how they perform.

Each experiment consists of three phases: _Setup_, _Populate_, and _Run_:

* **Setup**: Prepare the database and workload for the experiment (create tables, 
  indexes, download / generate data).
* **Populate**: Load the database with data, create indexes, etc.
* **Run**: Execute the workload against the database, measuring throughput, latency, 
  and other metrics.

VSB automatically runs these phases in sequence, and reports the results at the 
end; writing detailed results to `stats.json` file and displaying a summary.

### Supported databases

The following databases are currently supported by VSB:

* [Pinecone](vsb/databases/pinecone/README.md)
* [pgvector](vsb/databases/pgvector/README.md)

> [!TIP]
> The list of supported databases can also be shown from VSB by passing `help`
> to the `--database=` argument.

### Supported workloads

The following workloads are currently supported by VSB:

| Name        | Cardinality | Dimensions |    Metric | Description                                                                                                                                 |
|-------------|------------:|-----------:|----------:|---------------------------------------------------------------------------------------------------------------------------------------------|
| `mnist`     |      60,000 |        784 | euclidean | Images of handwritten digits from [MNIST](https://en.wikipedia.org/wiki/MNIST_database)                                                     |
| `nq768`     |   2,680,893 |        768 |    cosine | Natural language questions from [Google Research](https://ai.google.com/research/NaturalQuestions).                                         |
| `yfcc-10M`  |  10,000,000 |        192 | euclidean | Images from [Yahoo Flickr Creative Commons 100M](https://paperswithcode.com/dataset/yfcc100m) annotated with a "bag" of tags                |
| `cohere768` |  10,000,000 |        768 | euclidean | English wikipedia articles embedded with Cohere from [wikipedia-22-12](https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en) |

> [!TIP]
> The list of supported workloads can also be shown from VSB by passing `help`
> to the `--workload=` argument.

There are also smaller test workloads available for most workloads, e.g.
`mnist-test`, `nq768-test`. These are designed for basic sanity checking of a test
environment, however may not have correct ground-truth nearest neighbors and hence
should not be used for evaluating recall.

## Usage

Two parameters are required:

* `--database=<database>`: the database to run against.
* `--workload=<workload>`: the workload to execute.

Omitting the value for either database or workload will result in a list of 
available choices being displayed. 

Additional parameters may be specified to further configure the database 
and/or workload. For some databases these are required - for example to specify 
credentials or the index to connect to. Commonly used parameters are:

* `--requests_per_sec=<float>`: Cap the rate at which requests are issued to the 
  database.
* `--users=<int>`: Number of concurrent users (connections) to simulate.
* `--skip_populate`: Skip populating the database with data, just perform the Run 
  phase.

## What can I use VSB for?

VSB is designed to help you quickly measure how different workloads perform on a 
range of vector databases. It can be used to perform a range of tasks including:

* Compare the performance (throughput, latency, accuracy) of different vector databases.
* Benchmark the performance of a single database across different workloads.
* Evaluate the performance of a database under different conditions (e.g. different 
  data sizes, dimensions, metrics, access patterns).
* Understand the performance characteristics and identify bottlenecks of a database.
* Perform regression testing to ensure that changes to a database do not degrade performance.

### Measuring latency

VSB measures latency by sending a query to the database and measuring the duration 
between issuing the request and receiving the database's response.
This includes both the send/receive time and the time for the database to 
process the request. As such, latency is affected by the RTT between VSB 
client and the database, in addition to how long the database takes.

VSB records latency values for each request, then reported as percentiles 
when the workload completes. Live values (last 10s) are also displayed during the 
run for selected percentiles:
<p align="center" width="100%">
<img src=docs/images/vsb_example_live_metrics.png/>
</p>

Example: Run `yfcc-10M` (10M vectors, 192 dimensions, euclidean, metadata filtering) workload 
against Pinecone at 10 QPS:
```shell
vsb --database=pinecone --workload=yfcc-10M \
    --pinecone_index_name=<INDEX_NAME> --pinecone_api_key=<API_KEY> \
    --users=10 --requests_per_sec=10
```

#### Designing a good latency experiment

When measuring latency one typically wants to consider:

* **The rate** at which requests are issued (`--requests_per_sec`) so that
  neither the client machine nor the server (database) are saturated, as that
  would typically result in elevated latencies.

  Choose a request rate that is representative of the expected production
  workload.

* **The number of concurrent requests** (`--users`) to simulate. Most production 
  workloads will have multiple users (clients) issuing requests to the database
  concurrently, so it's important this is represented in the experiment.

* **What metrics to report**. Latency and throughput are "classic" measures of
  many database systems, however vector databases must also consider the
  quality of the responses to queries (e.g. what
  [recall](https://www.pinecone.io/learn/offline-evaluation/) is achieved at a
  given latency).
 
  Much like latency, it is important to consider the distribution of recall -
  a median (p50) recall of 80% may seem good, but if p10 recall is 0% then 10%
  of your queries are returning no relevent results.

### Measuring throughput

VSB measures throughput by calculating the number of responses which are
received over a given period of time. It maintains a running count of how many requests
have been performed over the course of the Run phase, and reports the overall rate 
when the experiment finishes. It also displays a live value (last 10s) during the run.

Example: Run nq-768 (2.68M vectors, 768 dimensions) workload  
against pgvector with multiple users and processes (to attempt to saturate it)
```shell
vsb --database=pgvector --workload=nq768 --users=10 --processes=10
```
#### Designing a good throughput experiment

Throughput experiments are typically trying to answer one of two questions:

1. Can the system handle the expected production workload - and if so how much will 
  it cost to run?
2. How far can the system scale within acceptable latency bounds?

In the first case throughput is an _input_ to the experiment - specify the expected 
workload via `--requests_per_sec=N`. In the second case throughput is an _output_ - 
we want to generate increasing amounts of load until the response time exceeds
our acceptable bounds - that is the maximum real-world throughput.

By default, VSB only simulates one user (`--users=1`), so the throughput is
effectively the reciprocal of the latency. Bear this in mind when trying to
measure throughput of a system - one would typically need to
increase the number of `--users` (and potentially `--processes`) to ensure there's
sufficient concurrent work given to the database system under test.

## Extending VSB

VSB has been designed to be extensible, to make it straightforward to add new  
databases workloads.

### Adding a new database

To add a new database to VSB you need to create a new module for your database
and implement 5 required methods, then register with VSB:

1. **Create a new module** in [`vsb/databases/`](vsb/databases/) for your database - e.g.
   for `mydb` create `vsb/databases/mydb/mydb.py`.
2. **Implement a Database class in this module** - inheriting from
   [`database.DB`](vsb/databases/base.py) and implementing the required methods:
    * `__init__()` - Set up the connection to your database, and any other required 
      initialisation.
    * `get_namespace()` - Returns a `Namespace` (table, sub-index) object to use for 
      the given namespace name. If the database doesn't support multiple namespaces 
      - or for an initial implemenation this can return just a single `Namespace` 
        object.
    * `get_batch_size()` - Returns the preferred size of record batch for the 
      populate phase. 
     
3. **Implement optional methods** if applicable to your database: 
   * `initialize_populate()` - Prepare the database for the populate phase - perhaps 
     create a table, or clear existing data.
   * `finalize_populate()` - Finalize the populate phase - perhaps create an index, 
     or wait for upserted data to be processed.
4. **Implement a Namespace class** in this module - inheriting from
   [`database.Namespace`](vsb/databases/base.py) and implementing the required
   methods:
    * `upsert_batch()` - Upsert the given batch of records into the namespace. 
    * `search()` - Perform a search for the given query vector.
5. **Register the database with VSB** by adding an entry to the `Database` enum in 
   [`vsb/databases/__init__.py`](vsb/databases/__init__.py), and updating `get_class()`.
6. (Optional) **Add database-specific command-line arguments** to `add_vsb_cmdline_args()` 
   method in [`vsb/cmdline_args.py`](vsb/cmdline_args.py) - for example for passing 
   credentials, connection parameters, index tunables.
7. (Optional) **Add Docker compose file** to [`docker`/](docker/) directory to 
   launch a local instance of the database to run against (only applicable to 
   locally running DBs).

That's it! You should now be able to run VSB against your database by specifying 
`--database=mydb` now.

#### Tips and tricks
* The existing database modules in VSB can be used as a reference for how to 
  implement a new database module - for example
  [`databases/pgvector`](vsb/databases/pgvector) for a locally-running DB, or
  [`database/pinecone`](vsb/databases/pinecone) for a cloud-hosted DB.
* Integration tests exist for each supported database, and can be run via `pytest` 
  to check that the module is working correctly. It is recommended you create
  a similar suite for your database.
* The `*-test` workloads are designed to be quick to run, and are a good starting 
  point for testing your database module. For example
  [workloads/mnist-test](vsb/workloads/mnist/mnist.py) is only 600 records and 20 
  queries, and should complete in a few seconds on most databases. 
  Once you have the test workloads working you can move on to the larger workloads.
* VSB uses standard Python logging, so you can use `logging` to output debug 
  information from your database module. The default emitted log level is `INFO`, 
  but this can be changed via `--loglevel=DEBUG`.

### Adding a new workload

To add a new workload to VSB you need to create a new module for your workload,
then register with VSB.

This can be any kind of workload - for example a synthetic workload, a real-world
dataset, or a workload based on a specific use-case. If the dataset already exists
in parquet format, you can use the
[`ParquetWorkload`](vsb/workloads/parquet_workload/parquet_workload.py) base class
to simplify this.
Otherwise, you'll have to implement the full [`VectorWorkload`](vsb/workloads/base.py)
base class.

#### Parquet-based workloads

VSB has support for loading  static datasets from Parquet files, assuming the 
files match the
[pinecone-datasets](https://github.com/pinecone-io/pinecone-datasets) schema.

1. **Create a new module** in [`vsb/workloads/`](vsb/workloads/) for your 
   workload - 
   e.g. for `my-workload` create `vsb/workloads/my-workload/my_workload.py`.
2. **Implement a Workload class** in this module - inheriting from
   [`parquet_workload.ParquetWorkload`](vsb/workloads/parquet_workload/parquet_workload.py)
   and implementing the required methods / properties:
    * `__init__()` - Call to the superclass constructor passing the dataset path - e.g:
      ```python
      class MyWorkload(ParquetWorkload):
          def __init__(self, name: str, cache_dir: str):
              super().__init__(name, "gs://bucket/my_workload", cache_dir=cache_dir)
       ```
    * `dimensions` - The dimensionality of the vectors in the dataset.
    * `metric` - The distance metric to use for the workload.
    * `record_count` - The number of records in the dataset.
    * `request_count` - The number of queries to perform.
3. **Register the workload with VSB** by adding an entry to the `Workload` enum in 
   [`vsb/workloads/__init__.py`](vsb/workloads/__init__.py), and updating `get_class()`.

All done. You should now be able to run this workload by specifying 
`--workload=my-workload`.

#### Other workloads

If the dataset is not in parquet format, or you want to have more control over the
operation of it, then you need to implement the
[`VectorWorkload`](vsb/workloads/base.py) base class:

1. **Create a new module** in [`vsb/workloads/`](vsb/workloads/) for your
   workload -
   e.g. for `my-workload` create `vsb/workloads/my-workload/my_workload.py`.
2. **Implement a Workload class** in this module - inheriting from
   [`base.VectorWorkload`](vsb/workloads/base.py) and
   implementing the required methods / properties:
    * `__init__()` - Whatever initialisation is needed for the workload.
    * `dimensions` - The dimensionality of the vectors in the dataset.
    * `metric` - The distance metric to use for the workload.
    * `record_count` - The number of records in the dataset.
    * `request_count` - The number of queries to perform.
    * `get_sample_record()` - Return a sample record from the dataset. (This is used by
      specific databases to calculate a suitable batch size for the populate phase.)
    * `get_record_batch_iter()` - Return an iterator over a batch of records to
      initially populate the database with.
    * `get_query_iter()` - Return an iterator over the queries a client should
      perform during the Run phase.
3. **Register the workload with VSB** by adding an entry to the `Workload` enum in
   [`vsb/workloads/__init__.py`](vsb/workloads/__init__.py), and updating `get_class()`.

All done. You should now be able to run this workload by specifying
`--workload=my-workload`.
