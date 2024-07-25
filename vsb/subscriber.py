from vsb import logger
from gevent.event import AsyncResult
import greenlet
import gevent
from locust.env import Environment
from locust.runners import WorkerRunner, MasterRunner

_results: dict[int, AsyncResult] = {}


class Subscriber:
    """A class that allows a Worker to pull a shared value managed by the Master.

    This has a very similar API and implementation to the locust_plugins.Distributor
    class, but only updates state when prompted by the Master. Workers can pull
    from the same subscriber however many times they want.
    """

    def __init__(self, environment: Environment, initial, name="subscriber"):
        """Register subscriber method handlers and set the initial value."""
        self.value = initial
        self.name = name
        self.runner = environment.runner
        if self.runner:
            # received on master
            def _request_data(environment, msg, **kwargs):
                """Master returns the current data value to the Worker."""
                # Run this in the background to avoid blocking locust's client_listener loop
                gevent.spawn(self._master_send, msg.data["gid"], msg.data["client_id"])

            # received on worker
            def _subscriber_on_res(environment: Environment, msg, **kwargs):
                _results[msg.data["gid"]].set(msg.data)

            self.runner.register_message(f"_{name}_request", _request_data)
            self.runner.register_message(f"_{name}_response", _subscriber_on_res)

    def _master_send(self, gid, client_id):
        self.runner.send_message(
            f"_{self.name}_response",
            {"value": self.value, "gid": gid},
            client_id=client_id,
        )

    def __call__(self):
        """Get data from master"""
        if not self.runner:  # no need to do anything clever if there is no runner
            assert self.value
            return self.value
        gid = greenlet.getcurrent().minimal_ident  # type: ignore

        if gid in _results:
            logger.warning("This user was already waiting for data. Strange.")

        _results[gid] = AsyncResult()
        self.runner.send_message(
            f"_{self.name}_request", {"gid": gid, "client_id": self.runner.client_id}
        )
        val = _results[gid].get()["value"]
        del _results[gid]
        return val

    def update(self, value):
        """Set data on master"""
        self.value = value
