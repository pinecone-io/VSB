# pgvector

This directory adds support running experiments against
[pgvector](https://github.com/pgvector/pgvector) - a PostgreSQL extension for
vector similarity search.

It supports both HNSW and IVFFlat index types, with configurable index creation
and search parameters.

## Usage

To run VSB against a pgvector instance, simply provide the `--database=pgvector`
flag:

```shell
vsb --database=pgvector --workload=mnist-test
```

This will use the default connection parameters for a pgvector instance running in
the local Docker container - see `docker/pgvector/` for the Docker compose file.

### Local Docker instance

VSB is setup to connect to a pgvector instance running in a local Docker container.
See `docker/pgvector/` for the Docker compose file.

You can connect to this instance via `psql` with the password `postgres`:

```shell
> psql --host=0.0.0.0 --port=5432 --dbname=postgres --username=postgres
```

### Connection parameters

To connect to a different instance, specify the appropriate connection parameters:

* `--pgvector_host`: Hostname of the pgvector instance
* `--pgvector_port`: Port of the pgvector instance
* `--pgvector_user`: Username to connect to the pgvector instance. Can also be set via
  the `VSB__PGVECTOR_USERNAME` environment variable.
* `--pgvector_password`: Password to connect to the pgvector instance. Can also be
  set the via the `VSB__PGVECTOR_PASSWORD` environment variable.

### Index creation

By default, the _hnsw_ index type is used. To use the _ivfflat_ index type, specify:

* `--pgvector_index_type=ivfflat`

The amount of memory to use when creating the index (`maintenance_work_mem`) can be set
via `--pgvector_maintenance_work_mem`. This defaults to 4GB (50% of the memory assigned
to the Docker container). This should generally be set to at least as large as
the index to avoid excessive index build times.

#### IVFFlat index type

By default, the number of lists the IVFFlat index is split into is set automatically
based on the suggested values([1]) for the workload:

> Choose an appropriate number of lists - a good place to start is `rows / 1000`
> for up to 1M rows and `sqrt(rows)` for over 1M rows

However this value can be overridden via:

* `--pgvector_ivfflat_lists=<int>`

### Index search

By default, the number of search candidates (`ef_search` for hnsw, `nprobes` for
ivfflat) is set automatically based on the recommended values([2],[1]) for the workload:

[1]: https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat

[2]: https://github.com/pgvector/pgvector?tab=readme-ov-file#why-are-there-less-results-for-a-query-after-adding-an-hnsw-index

* hnsw: `2 * top_k`
* ivfflat: `sqrt(lists)`

This can be overridden and an explicit value set via:
`--pgvector_search_candidates=<int>`.
