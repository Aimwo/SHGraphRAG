"""Explicit provider contract; never load dotenv or inherited API variables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from .common import ROOT, read_json, write_json
from .capacity_recovery import require_model_calls_allowed

CONFIG_PATH = ROOT / "experiment/config.json"
CREDENTIAL_ID = "user-supplied-DEEPSEEK_API_KEY"


def validate(config, *, ready=True):
    if config["provider"] != "deepseek-relay" or config["text_api"] != "openai-chat-completions":
        raise ValueError("This run only authorizes the selected DeepSeek relay")
    if config["credential_id"] != CREDENTIAL_ID:
        raise ValueError("Credential identity changed")
    if config["max_in_flight"] != 100:
        raise ValueError("The shared request ceiling must remain 100")
    embedding = config["embedding"]
    if any(embedding[k] != v for k, v in {
        "base_url": "http://127.0.0.1:11434", "model": "bge-m3:latest",
        "dimension": 1024, "max_in_flight": 4, "batch_size": 32, "truncate": None,
        "options": {"num_batch": 8192},
    }.items()):
        raise ValueError("Local BGE-M3 identity or batching changed")
    if config["sample_seed"] != 42 or config["sample_size"] != 1000:
        raise ValueError("Frozen sample settings changed")
    if ready:
        if (config["status"] != "ready" or not config.get("base_url")
                or not config.get("model")):
            raise RuntimeError(
                "Awaiting user-provided DeepSeek relay Base URL and exact API model ID; "
                "no credential was read and no API request was sent")
        url = urlsplit(config["base_url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("An explicit HTTPS relay Base URL is required")
        if not isinstance(config["model"], str) or any(c.isspace() for c in config["model"]):
            raise ValueError("An exact model ID without whitespace is required")
        if config["model"] not in config["allowed_returned_models"]:
            raise ValueError("Requested model must be in its explicit response allowlist")
    if set(config["generation_parameters"]) & {"model", "messages", "stream", "api_key"}:
        raise ValueError("Generation settings may not override provider identity")
    if config.get("proxy") not in (None, "http://127.0.0.1:6789"):
        raise ValueError("Only the explicitly selected per-client local proxy is supported")
    return config


def load_config(*, ready=True):
    return validate(read_json(CONFIG_PATH), ready=ready)


def read_authorized_key(config):
    import os
    validate(config)
    require_model_calls_allowed()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set your own authorized DEEPSEEK_API_KEY in this process")
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--returned-model", action="append", default=[])
    parser.add_argument("--generation-json", default="{}")
    parser.add_argument("--proxy", choices=["http://127.0.0.1:6789"])
    args = parser.parse_args()
    if (ROOT / "run/provenance.json").exists():
        raise RuntimeError("An actual run already exists; do not mutate its provider contract")
    config = load_config(ready=False)
    config.update(status="ready", base_url=args.base_url.rstrip("/"), model=args.model,
                  allowed_returned_models=sorted(set([args.model, *args.returned_model])),
                  generation_parameters=json.loads(args.generation_json), proxy=args.proxy)
    validate(config)
    write_json(CONFIG_PATH, config)
    print("Provider contract saved; no secret was read and no API request was made.")


if __name__ == "__main__":
    main()
