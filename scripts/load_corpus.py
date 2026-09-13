"""Load the oreo-data markdown corpus into a running doc-arag stack.

Usage (from the repository root):

    uv run python -m scripts.load_corpus --dry-run          # chunk report, no services
    uv run python -m scripts.load_corpus --recreate --only diagnostics/internet-check-procedure.md
    uv run python -m scripts.load_corpus                    # full corpus, skips loaded files
    uv run python -m scripts.load_corpus --force            # DELETE + re-upload every file

`domain` is the corpus sub-directory, `document_name` is the file name. Files
under `_*` directories and `EXCLUSIONS.md` are never loaded. Every loaded file
must grow the Weaviate object count, otherwise the run fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("load_corpus")

DEFAULT_CORPUS = Path("/Users/kskada/develop/oreo-data/corpus")
DEFAULT_API_URL = "http://localhost:8103"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
EXCLUDED_FILES = {"EXCLUSIONS.md"}
FILENAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\.md")
DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
POLL_INTERVAL_SECONDS = 2.0


def _bootstrap_env() -> None:
    """Placeholders so that importing docarag works with no services and no keys."""
    for key, value in (
        ("MINIO_ENDPOINT", "localhost:9000"),
        ("MINIO_ACCESS_KEY", "unused"),
        ("MINIO_SECRET_KEY", "unused"),
        ("MINIO_BUCKET", "unused"),
    ):
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    domain: str
    document_name: str

    @property
    def relative(self) -> str:
        return f"{self.domain}/{self.document_name}"


class CorpusError(Exception):
    """A corpus file cannot be loaded as-is."""


def discover_corpus_files(
    root: Path, only: list[str] | None = None
) -> list[CorpusFile]:
    """
    Walk `<root>/<domain>/*.md`, skipping `_*` directories and excluded files.

    Raises:
        CorpusError: If a file or directory name would break MinIO metadata or the
            domain slug validator (every offender is listed in the message)
    """
    if not root.is_dir():
        raise CorpusError(f"Corpus directory not found: {root}")

    files: list[CorpusFile] = []
    problems: list[str] = []
    for path in sorted(root.glob("*/*.md")):
        domain = path.parent.name
        if domain.startswith("_") or path.name in EXCLUDED_FILES:
            continue
        if not DOMAIN_PATTERN.match(domain):
            problems.append(f"{path}: directory '{domain}' is not a domain slug")
        if not FILENAME_PATTERN.fullmatch(path.name):
            problems.append(f"{path}: file name must be latin lowercase, ending in .md")
        files.append(CorpusFile(path=path, domain=domain, document_name=path.name))

    if problems:
        raise CorpusError("Corpus naming problems:\n  " + "\n  ".join(problems))

    if only:
        wanted = {item.strip("/") for item in only}
        files = [f for f in files if f.relative in wanted]
        missing = wanted - {f.relative for f in files}
        if missing:
            raise CorpusError(f"--only entries not found in corpus: {sorted(missing)}")

    return files


def chunk_report(files: list[CorpusFile]) -> int:
    """
    Parse every file locally and log a per-file chunk table.

    Returns:
        Total number of chunks across the corpus
    """
    from src.docarag.consts import MD_CHUNK_WARNING_THRESHOLD
    from src.docarag.services.parsers import parse_markdown
    from src.docarag.settings import settings

    logger.info(
        f"Chunking with md_chunk_size={settings.md_chunk_size}, "
        f"md_chunk_overlap={settings.md_chunk_overlap}, "
        f"warning threshold={MD_CHUNK_WARNING_THRESHOLD}"
    )
    header = (
        f"{'domain':<20} {'file':<34} {'chunks':>6} {'min':>5} {'median':>6} {'max':>5}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    total = 0
    oversized = 0
    domains: set[str] = set()
    for file in files:
        documents = parse_markdown(
            file.path.read_bytes(), settings.md_chunk_size, settings.md_chunk_overlap
        )
        sizes = [len(doc.page_content) for doc in documents]
        total += len(sizes)
        oversized += sum(1 for size in sizes if size > MD_CHUNK_WARNING_THRESHOLD)
        domains.add(file.domain)
        logger.info(
            f"{file.domain:<20} {file.document_name:<34} {len(sizes):>6} "
            f"{min(sizes):>5} {int(statistics.median(sizes)):>6} {max(sizes):>5}"
        )

    logger.info("-" * len(header))
    logger.info(
        f"files={len(files)} domains={len(domains)} chunks={total} "
        f"over_threshold={oversized}"
    )
    return total


# --- HTTP helpers -------------------------------------------------------------


def weaviate_total(client: httpx.Client, weaviate_url: str, collection: str) -> int:
    query = f"{{ Aggregate {{ {collection} {{ meta {{ count }} }} }} }}"
    response = client.post(f"{weaviate_url}/v1/graphql", json={"query": query})
    response.raise_for_status()
    payload = response.json()
    groups = payload.get("data", {}).get("Aggregate", {}).get(collection) or []
    return int(groups[0]["meta"]["count"]) if groups else 0


def weaviate_by_domain(
    client: httpx.Client, weaviate_url: str, collection: str
) -> dict[str, int]:
    query = (
        f'{{ Aggregate {{ {collection}(groupBy: ["domain"]) '
        f"{{ groupedBy {{ value }} meta {{ count }} }} }} }}"
    )
    response = client.post(f"{weaviate_url}/v1/graphql", json={"query": query})
    response.raise_for_status()
    groups = response.json().get("data", {}).get("Aggregate", {}).get(collection) or []
    return {str(g["groupedBy"]["value"]): int(g["meta"]["count"]) for g in groups}


def list_documents(client: httpx.Client, api_url: str) -> dict[str, str]:
    """Return `{filename: file_id}` for every uploaded document."""
    documents: dict[str, str] = {}
    page = 1
    while True:
        response = client.get(
            f"{api_url}/documents", params={"page": page, "page_size": 100}
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload["files"]:
            documents[item["filename"]] = item["file_id"]
        if page * payload["page_size"] >= payload["total"]:
            return documents
        page += 1


def upload_file(client: httpx.Client, api_url: str, file: CorpusFile) -> str:
    response = client.post(
        f"{api_url}/uploads",
        data={"document_name": file.document_name, "domain": file.domain},
        files={
            "document": (file.document_name, file.path.read_bytes(), "text/markdown")
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"upload failed {response.status_code}: {response.text[:300]}"
        )
    return str(response.json()["file_id"])


def embed_and_wait(
    client: httpx.Client, api_url: str, file_id: str, timeout: float
) -> dict[str, Any]:
    response = client.post(f"{api_url}/embeddings/{file_id}")
    if response.status_code >= 400:
        raise RuntimeError(
            f"embedding start failed {response.status_code}: {response.text[:300]}"
        )
    task_id = response.json()["task_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"{api_url}/tasks/{task_id}")
        task.raise_for_status()
        payload = task.json()
        if payload["status"] in ("completed", "failed"):
            return dict(payload)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"embedding task {task_id} did not finish within {timeout:.0f}s")


def delete_document(client: httpx.Client, api_url: str, file_id: str) -> None:
    response = client.delete(f"{api_url}/documents/{file_id}")
    if response.status_code >= 400:
        raise RuntimeError(
            f"delete failed {response.status_code}: {response.text[:300]}"
        )


def load_one(
    client: httpx.Client,
    api_url: str,
    weaviate_url: str,
    collection: str,
    file: CorpusFile,
    existing: dict[str, str],
    force: bool,
    timeout: float,
) -> bool:
    """
    Upload, embed and verify one file.

    Returns:
        True when the file was loaded (or skipped as already present), False on failure
    """
    if file.document_name in existing:
        if not force:
            logger.info(f"skip     {file.relative} (already uploaded)")
            return True
        logger.info(f"delete   {file.relative} ({existing[file.document_name]})")
        delete_document(client, api_url, existing[file.document_name])

    count_before = weaviate_total(client, weaviate_url, collection)
    file_id = upload_file(client, api_url, file)
    logger.info(f"upload   {file.relative} -> {file_id}")
    task = embed_and_wait(client, api_url, file_id, timeout)
    if task["status"] != "completed":
        logger.error(f"FAILED   {file.relative}: {task.get('message')}")
        return False
    count_after = weaviate_total(client, weaviate_url, collection)
    if count_after <= count_before:
        logger.error(
            f"FAILED   {file.relative}: object count did not grow "
            f"({count_before} -> {count_after}); task said {task.get('message')}"
        )
        return False
    logger.info(
        f"embedded {file.relative}: +{count_after - count_before} chunks "
        f"(total {count_after})"
    )
    return True


def recreate_collection(weaviate_url: str) -> None:
    """Drop and create the default collection directly, bypassing the API lifespan."""
    import asyncio

    parsed = urlparse(weaviate_url)
    os.environ["WEAVIATE_HOST"] = parsed.hostname or "localhost"
    os.environ["WEAVIATE_PORT"] = str(parsed.port or 8080)

    from src.docarag.services.vector_db import recreate_default_collection

    asyncio.run(recreate_default_collection())
    logger.info("collection recreated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--weaviate-url", default=DEFAULT_WEAVIATE_URL)
    parser.add_argument("--only", action="append", metavar="DOMAIN/FILE.md")
    parser.add_argument(
        "--dry-run", action="store_true", help="parse locally, print chunk report, exit"
    )
    parser.add_argument(
        "--force", action="store_true", help="delete and re-upload already loaded files"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and create the Weaviate collection first",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds to wait per file embedding",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    _bootstrap_env()

    from src.docarag.consts import DEFAULT_COLLECTION_NAME

    try:
        files = discover_corpus_files(args.corpus, args.only)
    except CorpusError as exc:
        logger.error(str(exc))
        return 1
    logger.info(f"{len(files)} corpus files under {args.corpus}")

    if args.dry_run:
        chunk_report(files)
        return 0

    if args.recreate:
        recreate_collection(args.weaviate_url)

    failures: list[str] = []
    with httpx.Client(timeout=60.0) as client:
        existing = list_documents(client, args.api_url)
        for file in files:
            try:
                ok = load_one(
                    client,
                    args.api_url,
                    args.weaviate_url,
                    DEFAULT_COLLECTION_NAME,
                    file,
                    existing,
                    args.force,
                    args.timeout,
                )
            except Exception as exc:  # one bad file must not hide the others
                logger.error(f"FAILED   {file.relative}: {exc}")
                ok = False
            if not ok:
                failures.append(file.relative)

        by_domain = weaviate_by_domain(
            client, args.weaviate_url, DEFAULT_COLLECTION_NAME
        )

    logger.info("chunks by domain:")
    for domain, count in sorted(by_domain.items()):
        logger.info(f"  {domain:<20} {count:>5}")
    logger.info(
        f"  {'total':<20} {sum(by_domain.values()):>5} in {len(by_domain)} domains"
    )

    if failures:
        logger.error(f"{len(failures)} file(s) failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
