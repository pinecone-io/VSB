"""
Create a Pinecone index for use in some later GitHub action. Takes input from
environment variables, then writes the output to the file named in
GITHUB_OUTPUT.
"""

import os
import random
import string
import datetime
from pinecone import Pinecone


def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception(f'Environment variable {name} is not set')
    return value


def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def write_gh_output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'{name}={value}', file=fh)


def main():
    pc = Pinecone(api_key=read_env_var('PINECONE_API_KEY'))
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat().replace(':', "-")
    index_name = read_env_var('NAME_PREFIX') + "--" + now + "--" + random_string(10)
    index_name = index_name.lower()
    environment = os.environ.get('ENVIRONMENT')
    if environment:
        spec = {
            'pod': {
                'environment': environment,
                'pod_type': 'p1.x1'
            }
        }
    else:
        spec = {
            'serverless': {
                'cloud': read_env_var('CLOUD'),
                'region': read_env_var('REGION'),
            }
        }

    pc.create_index(
        name=index_name,
        metric=read_env_var('METRIC'),
        dimension=int(read_env_var('DIMENSION')),
        spec=spec
    )
    index_host = 'https://' + pc.describe_index(index_name)['host']
    write_gh_output('index_name', index_name)
    write_gh_output('index_host', index_host)


if __name__ == '__main__':
    main()
