# Solr

This directory adds support for running experiments against
[Solr](https://solr.apache.org/) - an open-source search platform.

## To start a local Solr instance using Docker

To start, ensure you have Docker installed and running on your machine. Navigate to the `docker/solr` directory and run:

```shell
docker-compose up -d
```

## Seeding the default configset

When using a persistent Solr data volume, the built-in `_default` configset may not
be present under `/var/solr/data/configsets/_default`. To restore it, run:

```shell
docker exec solr bash -lc "mkdir -p /var/solr/data/configsets && \
  cp -r /opt/solr/server/solr/configsets/_default /var/solr/data/configsets/"
```

Then restart the container to make the configset available for core creation:

```shell
docker-compose restart solr
```

This ensures new cores can be created using the `_default` configset without errors.

This command will start Solr in a Docker container, mapping port 8983 on your host to port 8983 on the container. You can access the Solr admin interface by navigating to `http://localhost:8983/solr` in your web browser.

Invoke VSB with `--database=solr` and provide your Solr connection details.

```shell
vsb --database=solr --workload=mnist-test \
    --solr_url=http://localhost:8983/solr/
```

By default, this uses the workload name prefixed with 'vsb-' as the index name. The index will be created if it does not exist.

> [!TIP]
> The Solr URL and/or index name can also be passed via environment variables
> (`VSB__SOLR_URL` and `VSB__SOLR_INDEX_NAME` respectively).
