"""Lifecycle and ownership verification for this checkout's private Neo4j."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

import psutil

from reproduction.runtime import ROOT, utc_now, write_json
from reproduction.prepare_services import NEO4J_HOME

URI = "bolt://127.0.0.1:21887"
HTTP_PORT = 21474
BOLT_PORT = 21887
OWNER_FILE = ROOT / ".runtime/neo4j-instance.json"
JAVA_HOME = Path(os.environ.get("JAVA_HOME", ""))


def listening_pid():
    listeners = [
        item.pid for item in psutil.net_connections(kind="tcp")
        if item.status == psutil.CONN_LISTEN and item.laddr.port == BOLT_PORT
    ]
    if len(set(listeners)) > 1:
        raise RuntimeError("Multiple processes own the isolated Bolt port")
    return listeners[0] if listeners else None


def assert_owned_database():
    if os.environ.get("NEO4J_URI") != URI:
        raise RuntimeError("Refusing a graph connection outside the isolated URI")
    owner = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
    if Path(owner["home"]).resolve() != NEO4J_HOME.resolve() or owner["uri"] != URI:
        raise RuntimeError("Database ownership record identifies a different instance")
    pid = listening_pid()
    if pid is None or pid != owner["java_pid"]:
        raise RuntimeError("Owned database process is not listening")
    process = psutil.Process(pid)
    if abs(process.create_time() - owner["java_create_time"]) > 0.1:
        raise RuntimeError("Database PID was reused")
    if str(NEO4J_HOME).casefold() not in " ".join(process.cmdline()).casefold():
        raise RuntimeError("Listening Java process does not belong to this checkout")
    return owner


def configure_process_environment():
    os.environ.update({
        "NEO4J_URI": URI,
        "NEO4J_USERNAME": "neo4j",
        # This isolated loopback service has auth disabled; no DB secret exists.
        "NEO4J_PASSWORD": "isolated-local-no-auth",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_EMBEDDINGS_MODEL": "bge-m3:latest",
        "SHGRAPH_PROVIDER": "deepseek-relay",
        "SHGRAPH_GLOBAL_EXTRACTION": "1",
        "SHGRAPH_PARALLEL_DOC_MERGE": "1",
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "HF_DATASETS_CACHE": str(ROOT / ".cache/huggingface/datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "ANONYMIZED_TELEMETRY": "False",
        "NO_PROXY": "localhost,127.0.0.1,::1",
    })
    # No generation values are supplied by the public source. Explicitly omit
    # them for this run instead of inheriting unrelated application defaults.
    os.environ.pop("TEMPERATURE", None)
    os.environ.pop("MAX_TOKENS", None)


def start_database():
    existing = listening_pid()
    if existing is not None:
        configure_process_environment()
        return assert_owned_database()
    if not (NEO4J_HOME / "lib").exists():
        raise RuntimeError("Run reproduction.prepare_services before database startup")
    if not (JAVA_HOME / "bin/java.exe").exists():
        raise RuntimeError("Pinned local Java 17 installation is unavailable")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", HTTP_PORT)) == 0:
            raise RuntimeError("Isolated HTTP port is already in use")
    config = """server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=127.0.0.1:21887
server.http.enabled=true
server.http.listen_address=127.0.0.1:21474
server.https.enabled=false
dbms.security.auth_enabled=false
dbms.usage_report.enabled=false
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*
server.memory.heap.initial_size=1G
server.memory.heap.max_size=3G
server.memory.pagecache.size=1G
"""
    (NEO4J_HOME / "conf/neo4j.conf").write_text(config, encoding="utf-8")
    child_env = os.environ.copy()
    child_env.update({
        "JAVA_HOME": str(JAVA_HOME), "NEO4J_HOME": str(NEO4J_HOME),
        "NEO4J_CONF": str(NEO4J_HOME / "conf"),
        "NO_PROXY": "localhost,127.0.0.1,::1",
    })
    stdout_path = ROOT / ".runtime/neo4j-console.log"
    with stdout_path.open("ab") as stream:
        child = subprocess.Popen(
            [str(JAVA_HOME / "bin/java.exe"), "-cp", str(NEO4J_HOME / "lib/*"),
             "org.neo4j.server.startup.Neo4jBoot", "console"],
            cwd=NEO4J_HOME, env=child_env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            close_fds=True,
        )
    started = time.perf_counter()
    while time.perf_counter() - started < 90:
        pid = listening_pid()
        if pid is not None:
            process = psutil.Process(pid)
            owner = {
                "home": str(NEO4J_HOME), "uri": URI, "http_port": HTTP_PORT,
                "java_pid": pid, "java_create_time": process.create_time(),
                "launcher_pid": child.pid, "started_at": utc_now(),
                "java_home": str(JAVA_HOME), "auth_disabled_loopback_only": True,
                "configuration": config,
            }
            write_json(OWNER_FILE, owner)
            configure_process_environment()
            assert_owned_database()
            return owner
        if child.poll() is not None:
            raise RuntimeError(f"Neo4j startup exited {child.returncode}; inspect {stdout_path}")
        time.sleep(1)
    raise TimeoutError(f"Neo4j startup did not listen within 90 seconds; inspect {stdout_path}")


def stop_owned_database():
    """Stop only processes proven to belong to this checkout's Neo4j instance."""
    owner = assert_owned_database()
    processes = [psutil.Process(owner["java_pid"])]
    if owner["launcher_pid"] != owner["java_pid"] and psutil.pid_exists(owner["launcher_pid"]):
        launcher = psutil.Process(owner["launcher_pid"])
        if str(NEO4J_HOME).casefold() not in " ".join(launcher.cmdline()).casefold():
            raise RuntimeError("Refusing to stop an unowned launcher process")
        processes.append(launcher)
    for process in processes:
        process.terminate()
    _, remaining = psutil.wait_procs(processes, timeout=5)
    for process in remaining:
        # Re-check creation time and path immediately before a forced stop.
        if str(NEO4J_HOME).casefold() not in " ".join(process.cmdline()).casefold():
            raise RuntimeError("Process ownership changed during shutdown")
        process.kill()
    psutil.wait_procs(remaining, timeout=5)


if __name__ == "__main__":
    print(json.dumps(start_database(), indent=2))
