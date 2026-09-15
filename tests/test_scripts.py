"""Tests for the corpus loader and the control-question runner (no services)."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from scripts import load_corpus, run_control_questions


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics" / "internet-check.md").write_text(
        "---\ntitle: T\ndomain: diagnostics\n---\n\n# T\n\nintro\n\n## A\n\nbody a\n\n## B\n\nbody b\n",
        encoding="utf-8",
    )
    (tmp_path / "billing").mkdir()
    (tmp_path / "billing" / "overview.md").write_text(
        "# B\n\n## X\n\ntext\n", encoding="utf-8"
    )
    (tmp_path / "_raw").mkdir()
    (tmp_path / "_raw" / "draft.md").write_text("# draft\n", encoding="utf-8")
    (tmp_path / "EXCLUSIONS.md").write_text("# secrets\n", encoding="utf-8")
    return tmp_path


def test_discover_corpus_files_skips_raw_and_exclusions(corpus: Path):
    files = load_corpus.discover_corpus_files(corpus)

    assert [f.relative for f in files] == [
        "billing/overview.md",
        "diagnostics/internet-check.md",
    ]
    assert files[1].domain == "diagnostics"
    assert files[1].document_name == "internet-check.md"


def test_discover_corpus_files_only_filter(corpus: Path):
    files = load_corpus.discover_corpus_files(corpus, only=["billing/overview.md"])
    assert [f.relative for f in files] == ["billing/overview.md"]

    with pytest.raises(load_corpus.CorpusError, match="not found"):
        load_corpus.discover_corpus_files(corpus, only=["billing/missing.md"])


def test_discover_corpus_files_rejects_non_latin_names(corpus: Path):
    (corpus / "billing" / "обзор.md").write_text("# x\n", encoding="utf-8")

    with pytest.raises(load_corpus.CorpusError, match="latin"):
        load_corpus.discover_corpus_files(corpus)


def test_chunk_report_counts_sections(corpus: Path, caplog):
    load_corpus._bootstrap_env()
    files = load_corpus.discover_corpus_files(corpus)

    with caplog.at_level("INFO", logger="load_corpus"):
        total = load_corpus.chunk_report(files)

    # small sections are merged: billing "# B" + "## X" -> 1 chunk; diagnostics intro + A + B -> 1
    assert total == 2
    assert "files=2 domains=2 chunks=2" in caplog.text


def _api(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def test_load_one_skips_already_uploaded(corpus: Path):
    files = load_corpus.discover_corpus_files(corpus)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(500)

    with _api(handler) as client:
        ok = load_corpus.load_one(
            client,
            "http://api",
            "http://weaviate",
            "DefaultDocuments",
            files[0],
            {"overview.md": "id-1"},
            force=False,
            timeout=5,
        )

    assert ok is True
    assert calls == []


def test_load_one_force_deletes_then_reloads_and_checks_growth(corpus: Path):
    files = load_corpus.discover_corpus_files(corpus)
    calls: list[str] = []
    counts = iter([10, 13])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v1/graphql":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "Aggregate": {
                            "DefaultDocuments": [{"meta": {"count": next(counts)}}]
                        }
                    }
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"status": "deleted"})
        if request.url.path == "/uploads":
            body = request.content
            assert b'name="domain"' in body and b"billing" in body
            return httpx.Response(200, json={"file_id": "new-id"})
        if request.url.path == "/embeddings/new-id":
            return httpx.Response(200, json={"task_id": "t1"})
        if request.url.path == "/tasks/t1":
            return httpx.Response(200, json={"status": "completed", "message": "ok"})
        return httpx.Response(404)

    with _api(handler) as client:
        ok = load_corpus.load_one(
            client,
            "http://api",
            "http://weaviate",
            "DefaultDocuments",
            files[0],
            {"overview.md": "old-id"},
            force=True,
            timeout=5,
        )

    assert ok is True
    assert calls[0] == "DELETE /documents/old-id"
    assert "POST /uploads" in calls and "POST /embeddings/new-id" in calls


def test_load_one_fails_when_count_does_not_grow(corpus: Path, caplog):
    files = load_corpus.discover_corpus_files(corpus)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/graphql":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "Aggregate": {"DefaultDocuments": [{"meta": {"count": 10}}]}
                    }
                },
            )
        if request.url.path == "/uploads":
            return httpx.Response(200, json={"file_id": "new-id"})
        if request.url.path == "/embeddings/new-id":
            return httpx.Response(200, json={"task_id": "t1"})
        if request.url.path == "/tasks/t1":
            return httpx.Response(200, json={"status": "completed", "message": "ok"})
        return httpx.Response(404)

    with _api(handler) as client, caplog.at_level("ERROR", logger="load_corpus"):
        ok = load_corpus.load_one(
            client,
            "http://api",
            "http://weaviate",
            "DefaultDocuments",
            files[0],
            {},
            force=False,
            timeout=5,
        )

    assert ok is False
    assert "did not grow" in caplog.text


def test_weaviate_by_domain_parses_groups():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "groupBy" in json.loads(request.content)["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "Aggregate": {
                        "DefaultDocuments": [
                            {"groupedBy": {"value": "billing"}, "meta": {"count": 20}},
                            {
                                "groupedBy": {"value": "diagnostics"},
                                "meta": {"count": 22},
                            },
                        ]
                    }
                }
            },
        )

    with _api(handler) as client:
        assert load_corpus.weaviate_by_domain(
            client, "http://w", "DefaultDocuments"
        ) == {
            "billing": 20,
            "diagnostics": 22,
        }


def test_check_answer_flags_prices_names_and_low_confidence():
    good = {"answer": "Ответ по регламенту", "confidence": 0.9, "sources_used": 3}
    assert run_control_questions.check_answer({"expect": "positive"}, good, []) == []

    priced = {**good, "answer": "Тариф стоит 1 500 руб. в месяц"}
    assert "answer contains a price" in run_control_questions.check_answer(
        {"expect": "no-prices"}, priced, []
    )

    named = {**good, "answer": "Расчётами занимается Иванова"}
    assert any(
        "blacklisted" in f
        for f in run_control_questions.check_answer(
            {"expect": "no-names"}, named, ["Иванова"]
        )
    )

    weak = {**good, "confidence": 0.4, "sources_used": 0}
    failures = run_control_questions.check_answer({"expect": "positive"}, weak, [])
    assert "sources_used=0" in failures and "confidence<0.7" in failures


def test_control_run_writes_report(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "llm_provider": "openai",
                    "llm_model": "m",
                    "embedding_model": "e",
                    "reranker_provider": "none",
                },
            )
        if request.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "query": "q",
                    "answer": "Ответ",
                    "confidence": 0.8,
                    "iterations": 1,
                    "sources_used": 2,
                },
            )
        return httpx.Response(404)

    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "x1",
                    "domain": "billing",
                    "expect": "positive",
                    "question": "Вопрос?",
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.md"

    original = httpx.Client
    with patch(
        "scripts.run_control_questions.httpx.Client",
        side_effect=lambda **kw: original(
            transport=httpx.MockTransport(handler), timeout=5
        ),
    ):
        code = run_control_questions.main(
            [
                "--questions",
                str(questions),
                "--out",
                str(out),
                "--blacklist",
                str(tmp_path / "none.txt"),
            ]
        )

    assert code == 0
    assert "1/1 passed" in out.read_text(encoding="utf-8")
    assert out.with_suffix(".jsonl").exists()
