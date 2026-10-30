"""Loaded by every Python process of pgolf-smoke's run (its folder is put on PYTHONPATH): records, with the time,
each file the run opens under the data root and each network access Python makes to another machine, so smoke.py
can tell whether training data was read after training ended or the network was used. Writes one JSON line per
event to $PGOLF_AUDIT_LOG.<pid>. It only observes: stdout is wrapped to time the training lines as they are
written, and its text passes through unchanged. Not used by the benchmark."""

import json
import os
import re
import socket
import sys
import time

_LOG = os.environ.get("PGOLF_AUDIT_LOG")
_ROOT = os.environ.get("PGOLF_DATA_ROOT", "")
_LOCAL_PREFIXES = ("127.", "0.0.0.0", "::1", "::", "localhost", "fe80:")


def _install() -> None:
    f = open(f"{_LOG}.{os.getpid()}", "a", buffering=1)
    me = {socket.gethostname(), socket.getfqdn(), ""}

    def local(host: str) -> bool:
        host = str(host)
        if host in me or host.startswith(_LOCAL_PREFIXES):
            return True
        # this machine's own addresses (torchrun's rendezvous can use them)
        try:
            return host in {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None)}
        except OSError:
            return False

    busy = [False]

    def hook(event: str, args: tuple) -> None:
        if busy[0]:
            return
        busy[0] = True
        try:
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                path = os.fsdecode(args[0])
                if _ROOT and path.startswith(_ROOT) or "fineweb_" in path:
                    f.write(json.dumps(["open", time.time(), path]) + "\n")
            elif event == "socket.connect" and len(args) > 1:
                addr = args[1]
                host = addr[0] if isinstance(addr, tuple) else addr
                if isinstance(host, str) and not host.startswith("/") and not local(host):
                    f.write(json.dumps(["connect", time.time(), str(addr)]) + "\n")
            elif event == "socket.getaddrinfo" and args and not local(args[0]):
                f.write(json.dumps(["resolve", time.time(), str(args[0])]) + "\n")
            elif event == "urllib.Request" and args:
                f.write(json.dumps(["urllib", time.time(), str(args[0])]) + "\n")
        except Exception:  # noqa: BLE001  (auditing must never break the run)
            pass
        finally:
            busy[0] = False

    sys.addaudithook(hook)

    # Training lines (`step:...train_time:` and `stopping_early:`), timed in this process, so a data file opened after
    # the last one is known to come after it (the eval's own timestamps of the same lines arrive a little later).
    train_line = re.compile(r"^(?:step:\d+/\d+ .*train_time:|stopping_early:)", re.M)

    class Stdout:
        def __init__(self, inner):
            self._inner = inner

        def write(self, text):
            try:
                if isinstance(text, str) and train_line.search(text):
                    f.write(json.dumps(["train_line", time.time(), ""]) + "\n")
            except Exception:  # noqa: BLE001
                pass
            return self._inner.write(text)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    sys.stdout = Stdout(sys.stdout)


if _LOG:
    try:
        _install()
    except Exception:  # noqa: BLE001
        pass
