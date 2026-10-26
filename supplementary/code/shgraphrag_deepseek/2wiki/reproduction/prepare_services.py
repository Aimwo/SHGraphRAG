"""Download pinned public Neo4j artifacts into this isolated checkout only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
DOWNLOADS = RUNTIME / "downloads"
NEO4J_URL = "https://dist.neo4j.org/neo4j-community-5.26.0-windows.zip"
GDS_URL = "https://github.com/neo4j/graph-data-science/releases/download/2.13.4/neo4j-graph-data-science-2.13.4.jar"
APOC_URL = "https://github.com/neo4j/apoc/releases/download/5.26.0/apoc-5.26.0-core.jar"
NEO4J_HOME = RUNTIME / "neo4j-community-5.26.0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opener(proxy):
    # Only the public Neo4j CDN needs this command-scoped route.
    selected = os.environ.get("SHGRAPHRAG_DOWNLOAD_PROXY") if proxy else None
    proxies = {"http": selected, "https": selected} if selected else {}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def download(url, proxy=False):
    path = DOWNLOADS / url.rsplit("/", 1)[-1]
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = None
        for attempt in range(3):
            try:
                response = opener(proxy).open(request, timeout=45)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        with response, temporary.open("wb") as stream:
            total = 0
            last_reported = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                total += len(block)
                if total - last_reported >= 16 * 1024 * 1024:
                    print(f"{path.name}: {total // (1024 * 1024)} MiB", flush=True)
                    last_reported = total
        temporary.replace(path)
    return path, {
        "url": url, "file": str(path), "bytes": path.stat().st_size,
        "sha256": sha256(path), "command_scoped_proxy": proxy,
    }


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    records = []
    neo4j, record = download(NEO4J_URL, proxy=True)
    # The checksum endpoint is public metadata; never read any API credentials.
    try:
        checksum_request = urllib.request.Request(
            NEO4J_URL + ".sha256", headers={"User-Agent": "Mozilla/5.0"}
        )
        with opener(True).open(checksum_request, timeout=20) as response:
            expected = response.read().decode().strip().split()[0].lower()
        if len(expected) != 64 or expected != record["sha256"]:
            raise ValueError("Neo4j distribution checksum mismatch")
        record["official_sha256_verified"] = True
    except (urllib.error.URLError, TimeoutError) as error:
        record["official_checksum_endpoint_error"] = str(error)
        record["official_sha256_verified"] = False
    records.append(record)
    if not (NEO4J_HOME / "lib").exists():
        with zipfile.ZipFile(neo4j) as archive:
            target_root = RUNTIME.resolve()
            for member in archive.infolist():
                resolved = (RUNTIME / member.filename).resolve()
                if not resolved.is_relative_to(target_root):
                    raise ValueError("Unsafe distribution archive path")
            archive.extractall(RUNTIME)
    for url in (GDS_URL, APOC_URL):
        path, record = download(url)
        records.append(record)
        destination = NEO4J_HOME / "plugins" / path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
        elif sha256(destination) != record["sha256"]:
            raise ValueError("Plugin differs from pinned download")
    (RUNTIME / "download_inventory.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ready": True, "neo4j_home": str(NEO4J_HOME), "downloads": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
