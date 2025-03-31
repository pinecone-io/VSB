import os
import sys
from unittest.mock import patch
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vsb.randomized_pacer import RandomizedPacer
import pytest


class TestRandomizedPacer:
    def test_init(self):
        pacer = RandomizedPacer(1.0)
        assert pacer.target_rps == 1.0

    def test_zero_rps(self):
        # When target_rps is 0, wait_time should return 0
        pacer = RandomizedPacer(0.0)
        assert pacer.wait_time() == 0

    @patch("time.time")
    def test_behind_schedule(self, mock_time):
        # Test when we're behind schedule
        mock_time.return_value = 100.0
        pacer = RandomizedPacer(1.0)

        # Now we're 2 seconds behind
        mock_time.return_value = 102.0

        with patch("random.random", return_value=0.5):
            wait = pacer.wait_time()
            assert wait == 0.0  # Should return 0 when behind schedule

    def test_long_term_rate_stability(self):
        # Test that over many calls, the average rate is maintained
        # This test mocks time to make it deterministic
        with patch("time.time") as mock_time:
            current_time = 1000.0
            mock_time.return_value = current_time

            pacer = RandomizedPacer(10.0)
            wait_times = []

            # Simulate many requests with mocked time
            samples = 1000  # Large sample for statistical confidence
            for _ in range(samples):
                wait = pacer.wait_time()
                wait_times.append(wait)
                current_time += wait
                mock_time.return_value = current_time

            # Calculate effective rate
            total_time = sum(wait_times)
            effective_rps = samples / total_time if total_time > 0 else float("inf")

            # Rate should be close to target (allowing for some variation)
            assert (
                9.5 <= effective_rps <= 10.5
            ), f"Effective rate {effective_rps} is too far from target 10.0"

    def test_wait_time_distribution(self):
        """
        Test that wait times are properly distributed and maintain the target rate.

        This test validates:
        1. Wait times are properly randomized (not all the same)
        2. The overall request rate matches the target rate
        3. The distribution has expected statistical properties
        """
        with patch("time.time") as mock_time:
            current_time = 1000.0
            mock_time.return_value = current_time

            # Create a pacer with 1 request per second
            pacer = RandomizedPacer(1.0)

            # Collect wait times and timestamps
            wait_times = []
            timestamps = [current_time]  # Start with initial time

            for _ in range(1000):
                wait = pacer.wait_time()
                wait_times.append(wait)

                # Simulate time passing exactly by the wait duration
                current_time += wait
                mock_time.return_value = current_time
                timestamps.append(current_time)

        # Verify wait times are non-negative
        assert min(wait_times) >= 0, "Wait times should not be negative"

        # Verify wait times are randomized (have variance)
        assert statistics.stdev(wait_times) > 0.05, "Wait times should be distributed"

        # Verify the mean wait time is close to the expected interval
        # With 1000 samples, it should be very close to the target interval of 1 second
        mean_wait = statistics.mean(wait_times)
        assert (
            0.95 <= mean_wait <= 1.05
        ), f"Mean wait time {mean_wait} should be close to 1.0 second"

        # Verify the request rate over the entire period matches the target
        total_time = timestamps[-1] - timestamps[0]
        req_per_sec = 1000 / total_time
        assert (
            0.95 <= req_per_sec <= 1.05
        ), f"Request rate {req_per_sec} should be close to 1.0"

        # Verify successive wait times aren't too strongly correlated
        # (which would indicate poor randomization)
        successive_diffs = [
            abs(wait_times[i] - wait_times[i - 1]) for i in range(1, len(wait_times))
        ]
        avg_diff = statistics.mean(successive_diffs)
        assert avg_diff > 0.1, "Wait times should vary between successive requests"
