# VSB: Vector Search Bench

<p align="center" width="100%">
   <img src=docs/images/splash.jpg width="320px"/>>
</p>

**VSB** is a benchmarking suite for Vector Search.

## Quickstart

### Requirements
* Python >= 3.11

### Install
1. Clone this repo:
   ```shell
   git clone https://github.com/pinecone-io/VSB.git
   ```
2. Use Poetry to install dependencies
   ```shell
   cd VSB
   pip3 install poetry
   poetry install
   ``` 
3. Activate environment containing dependencies
   ```shell
   poetry shell
   ```

### Run
   ```shell
   ./vsb.py --help
   ```
This will print a message showing how to run, including the required arguments.
For example to run the test variant of MNIST against pinecone
```shell
./vsb.py --database=pinecone --workload=mnist-test \
    --api_key=<API_KEY> \
    --index_name=<INDEX_NAME>
```
Where `--api_key` specifies the Pinecone API key to use and `--index_name` specifies the name of the index to connect to.

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