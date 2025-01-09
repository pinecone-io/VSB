import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vsb.workloads.synthetic_workload.synthetic_workload import (
    SyntheticProportionalWorkload,
)
import pytest
import numpy as np


class TestSyntheticMetadata:
    @pytest.fixture
    def rng(self):
        """Provides a random generator instance for testing."""
        return np.random.default_rng(42)

    def test_string_generator(self, rng):
        """Test that the string generator produces fixed-length alphanumeric strings."""
        generator = SyntheticProportionalWorkload.make_string_generator(10)
        result = generator(rng)
        assert isinstance(result, str)
        assert len(result) == 10
        assert all(c.isalnum() for c in result)

    def test_list_generator(self, rng):
        """Test that the list generator produces a list of n strings of length m."""
        generator = SyntheticProportionalWorkload.make_list_generator(5, 3)
        result = generator(rng)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(s, str) and len(s) == 5 for s in result)

    def test_numeric_generator(self, rng):
        """Test that the numeric generator produces an integer with the expected digit count."""
        generator = SyntheticProportionalWorkload.make_numeric_generator(5)
        result = generator(rng)
        assert isinstance(result, int)
        assert result < 10000  # Ensures a 5-digit number

    def test_boolean_generator(self, rng):
        """Test that the boolean generator produces True or False."""
        generator = SyntheticProportionalWorkload.make_boolean_generator()
        result = generator(rng)
        assert isinstance(result, bool)

    def test_parse_synthetic_metadata_template(self, rng):
        """Test full metadata parsing and generation."""
        metadata_template = ["id:10n", "tags:5s10l", "flag:b", "username:8s"]
        generators = SyntheticProportionalWorkload.parse_synthetic_metadata_template(
            metadata_template
        )

        # Ensure correct generator keys exist
        assert set(generators.keys()) == {"id", "tags", "flag", "username"}

        # Generate metadata
        metadata = {key: gen(rng) for key, gen in generators.items()}

        # Validate "id" as a numeric value with 10 digits
        assert isinstance(metadata["id"], int)
        assert metadata["id"] < 10**10  # Ensures a 10-digit number

        # Validate "tags" as a list of 10 strings, each of length 5
        assert isinstance(metadata["tags"], list)
        assert len(metadata["tags"]) == 10
        assert all(isinstance(tag, str) and len(tag) == 5 for tag in metadata["tags"])

        # Validate "flag" as a boolean
        assert isinstance(metadata["flag"], bool)

        # Validate "username" as a string of length 8
        assert isinstance(metadata["username"], str)
        assert len(metadata["username"]) == 8
        assert all(c.isalnum() for c in metadata["username"])  # Ensure alphanumeric
