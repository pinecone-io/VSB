import os

if os.name == "nt":
    import msvcrt

    def portable_lock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def portable_unlock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def portable_lock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def portable_unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class Locker:
    """A context manager that uses OS-native file locks to ensure
    inter-process synchronization. This is used to perform operations
    exclusively once per machine, such as dataset downloading, on
    distributed Locust runs without monkeypatching the Locust
    multiprocessing implementation."""

    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if os.path.exists(filepath):
            os.remove(filepath)

    def __del__(self):
        # clean up lock file
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass

    def __enter__(self):
        self.fp = open(self.filepath, "w+")
        portable_lock(self.fp)

    def __exit__(self, _type, value, tb):
        portable_unlock(self.fp)
        self.fp.close()
