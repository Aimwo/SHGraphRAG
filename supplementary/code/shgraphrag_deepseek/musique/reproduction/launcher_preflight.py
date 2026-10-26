"""Check pinned public configuration and controller identity without API calls."""

import json

import psutil

from reproduction.campaign import DEFAULT_RUN
from reproduction.relay_config import load_relay_config


def main():
    load_relay_config()
    path = DEFAULT_RUN / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pid = state.get("pid")
    if pid:
        try:
            process = psutil.Process(pid)
            if (
                process.is_running()
                and abs(process.create_time() - state.get("process_create_time", 0)) < 0.1
            ):
                raise RuntimeError("This exact experiment is already running")
        except psutil.NoSuchProcess:
            pass
    print("Pinned relay configuration and inactive controller verified")


if __name__ == "__main__":
    main()
