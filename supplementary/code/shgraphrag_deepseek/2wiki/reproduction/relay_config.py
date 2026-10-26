"""Public reproduction configuration; explicit environment credential only."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "reproduction/relay_config.json"
EXPECTED_CREDENTIAL_ID = "user-supplied-DEEPSEEK_API_KEY"


def load_relay_config(require_ready=True):
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if config["provider"] != "deepseek-relay" or config["credential_id"] != EXPECTED_CREDENTIAL_ID:
        raise ValueError("Configured provider identity differs")
    if config["max_in_flight"] not in (64, 100):
        raise ValueError("Unexpected historical request ceiling")
    if require_ready:
        if config["status"] != "ready" or not config.get("base_url") or not config.get("model"):
            raise RuntimeError("Provide a ready Base URL and exact model before running")
        url = urlsplit(config["base_url"])
        if (config["status"] != "ready" or url.scheme != "https"
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("An explicit HTTPS provider is required")
        if config["model"] not in config["allowed_returned_models"]:
            raise ValueError("Model must be in the returned-model allowlist")
    return config


def config_identity(config):
    public = {key: value for key, value in config.items() if key != "authorization"}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def read_authorized_key(config):
    if os.environ.get("SHGRAPHRAG_ALLOW_MODEL_API") != "1":
        raise RuntimeError("Model calls require explicit opt-in for this fresh run")
    load_relay_config()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set your own authorized DEEPSEEK_API_KEY in this process")
    return key


def resume_config_variants(config):
    """Allow only the approved 64-to-100 scheduling change, with all else exact."""
    variants = [(config, None)]
    if config["max_in_flight"] == 100:
        variants.append((
            {**config, "max_in_flight": 64},
            "user-approved-request-capacity64-to100-20261025",
        ))
    return variants
