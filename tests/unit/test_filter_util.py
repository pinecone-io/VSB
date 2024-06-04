import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vsb.databases.pgvector.filter_util import FilterUtil
import pytest


class TestFilterUtil:
    def test_set_membership(self):
        json_str = {"tags": "28"}
        expected_sql = """WHERE metadata @> \'{"tags": ["28"]}\'"""
        assert FilterUtil.to_sql(json_str) == expected_sql

    def test_set_membership_and(self):
        json_str = {"$and": [{"tags": "5692"}, {"tags": "2212"}]}
        expected_sql = """WHERE metadata @> \'{"tags": ["5692"]}\' AND metadata @> \'{"tags": ["2212"]}\'"""
        assert FilterUtil.to_sql(json_str) == expected_sql

    def test_set_membership_and_and(self):
        json_str = {"$and": [{"tags": "1"}, {"$and": [{"tags": "2"}, {"tags": "3"}]}]}
        expected_sql = """WHERE metadata @> \'{"tags": ["1"]}\' AND metadata @> \'{"tags": ["2"]}\' AND metadata @> \'{"tags": ["3"]}\'"""
        assert FilterUtil.to_sql(json_str) == expected_sql
