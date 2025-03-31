import time
import random


class RandomizedPacer:
    """
    Maintains a consistent average request rate while randomizing exact request timing
    to avoid coordination between multiple instances.
    """

    def __init__(self, target_rps):
        """
        Initialize pacing controller with target requests per second.
        """
        self.target_rps = target_rps
        self.next_expected_time = time.time()

    def wait_time(self):
        """
        Calculate wait time for next request, introducing randomization while
        maintaining the target throughput over time.

        Returns:
            float: Time to wait in seconds before next request
        """
        now = time.time()

        # Calculate ideal time between requests
        interval = 1.0 / self.target_rps if self.target_rps > 0 else 0

        # If we're behind schedule, don't wait
        if now >= self.next_expected_time:
            # Schedule the next request one interval from now
            self.next_expected_time = now + interval
            return 0

        # Calculate wait time to next expected request (assuming exact target
        # rate)
        wait = self.next_expected_time - now

        # Introduce randomization: instead of waiting the full time,
        # we'll wait a random amount between 0 and 2*wait - i.e. on average
        # we wait for `wait` seconds, maintaining the target rate.
        random_factor = random.random() * 2.0
        randomized_wait = wait * random_factor

        # Schedule the next expected request time:
        # 1. Move forward by one full interval
        # 2. Adjust for the randomization to maintain the average rate
        adjustment = wait - randomized_wait
        self.next_expected_time = self.next_expected_time + interval - adjustment

        return max(randomized_wait, 0)
