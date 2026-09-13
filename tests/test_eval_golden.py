"""Tests for the golden-set validator and scorer (no services)."""

import json
from pathlib import Path

import pytest

from scripts import eval_golden


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "diagnostics").mkdir(parents=True)
    (root / "diagnostics" / "check.md").write_text(
        "# Проверка\n\n## Шаг 1. Сессия\n\nМодуль PPPoE, вкладка «Логи».\n\n## Шаг 2. Лимит\n\nДоверительный платёж 3 дня.\n",
        encoding="utf-8",
    )
    return root


def _record(**overrides):
    base = {
        "id": "diag-001",
        "domain": "diagnostics",
        "source": "diagnostics/check.md",
        "section": "Шаг 1. Сессия",
        "type": "factual",
        "question": "Где смотреть сессию?",
        "reference_answer": "В модуле PPPoE на вкладке Логи.",
        "must_include": ["pppoe", "логи"],
        "must_not_include": ["руб"],
        "_origin": "parts/x.jsonl:1",
    }
    base.update(overrides)
    return base


def test_validate_accepts_consistent_record(corpus: Path):
    assert eval_golden.validate_records([_record()], corpus) == []


def test_validate_reports_missing_fact_section_domain_and_duplicates(corpus: Path):
    records = [
        _record(),
        _record(
            id="diag-001",
            must_include=["нет такого"],
            section="Шаг 9",
            domain="billing",
        ),
        _record(id="diag-003", type="weird", question="no cyrillic"),
    ]

    problems = eval_golden.validate_records(records, corpus)

    joined = "\n".join(problems)
    assert "duplicate id diag-001" in joined
    assert "must_include 'нет такого' not found" in joined
    assert "section not a heading" in joined
    assert "domain 'billing' != directory 'diagnostics'" in joined
    assert "unknown type 'weird'" in joined
    assert "not in Russian" in joined


def test_validate_reports_missing_source(corpus: Path):
    problems = eval_golden.validate_records(
        [_record(source="diagnostics/none.md")], corpus
    )
    assert problems == [
        "parts/x.jsonl:1 (diag-001): source not found: diagnostics/none.md"
    ]


def test_load_parts_and_write_merged(tmp_path: Path, corpus: Path):
    golden = tmp_path / "golden"
    (golden / "parts").mkdir(parents=True)
    rec = {k: v for k, v in _record().items() if not k.startswith("_")}
    (golden / "parts" / "a.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n\n", encoding="utf-8"
    )

    records = eval_golden.load_parts(golden)
    assert len(records) == 1 and records[0]["_origin"] == "a.jsonl:1"

    out = eval_golden.write_merged(records, golden)
    merged = json.loads(out.read_text(encoding="utf-8").strip())
    assert merged == rec

    assert eval_golden.load_records(golden)[0]["id"] == "diag-001"


def test_score_record_hits_facts_and_forbidden():
    record = _record()
    result = {
        "answer": "Смотрим модуль PPPoE, вкладка Логи. Стоит 100 руб.",
        "confidence": 0.8,
        "sources_used": 3,
        "sources": [
            {"document_name": "other.md", "domain": "billing"},
            {"document_name": "check.md", "domain": "diagnostics"},
        ],
    }

    scores = eval_golden.score_record(record, result, k=5)

    assert scores["doc_hit"] is True
    assert scores["domain_hit"] is True
    assert scores["fact_coverage"] == 1.0
    assert scores["forbidden"] == ["100 руб"]
    assert scores["russian"] is True

    scores_k1 = eval_golden.score_record(record, result, k=1)
    assert scores_k1["doc_hit"] is False and scores_k1["domain_hit"] is False


def test_score_record_partial_coverage():
    scores = eval_golden.score_record(
        _record(), {"answer": "Только логи", "sources": []}, k=5
    )
    assert scores["fact_coverage"] == 0.5
    assert scores["facts_missing"] == ["pppoe"]


def test_parse_judge_reply_json_and_fallback():
    assert eval_golden.parse_judge_reply('{"score": 4, "reason": "ok"}') == (4, "ok")
    assert eval_golden.parse_judge_reply(
        'Sure:\n```json\n{"score": 2, "reason": "off"}\n```'
    ) == (2, "off")
    assert eval_golden.parse_judge_reply("I would give it 3 out of 5")[0] == 3
    assert eval_golden.parse_judge_reply("no idea")[0] is None


def test_aggregate_and_render():
    rows = [
        {
            **_record(),
            "doc_hit": True,
            "domain_hit": True,
            "fact_coverage": 1.0,
            "facts_missing": [],
            "forbidden": [],
            "confidence": 0.9,
            "answer": "a",
            "judge_score": 5,
            "judge_reason": "good",
        },
        {
            **_record(id="diag-002", type="negative"),
            "doc_hit": False,
            "domain_hit": True,
            "fact_coverage": 0.5,
            "facts_missing": ["pppoe"],
            "forbidden": ["руб"],
            "confidence": 0.5,
            "answer": "b",
            "judge_score": None,
            "judge_reason": "",
        },
    ]
    stats = eval_golden.aggregate(rows)
    assert stats["n"] == 2 and stats["doc_hit"] == 0.5 and stats["forbidden"] == 1
    assert stats["judge"] == 5

    report = eval_golden.render_markdown({"run_at": "now", "llm_model": "m"}, rows, k=5)
    assert "| **all** | 2 | 50% | 100% | 75% | 1 |" in report
    assert "type=negative" in report and "❌ руб" in report


def test_judge_answer_merges_extra_body_and_parses_score():
    import httpx

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"score": 4, "reason": "minor omission"}'}}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        score, reason = eval_golden.judge_answer(
            client,
            "http://judge.test/v1",
            "k",
            "judge-model",
            _record(),
            "ответ",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    assert (score, reason) == (4, "minor omission")
    assert seen["url"] == "http://judge.test/v1/chat/completions"
    assert seen["body"]["model"] == "judge-model"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["body"]["temperature"] == 0


def test_currency_forbidden_requires_an_amount():
    record = _record(must_not_include=["руб", "₽", "Иванова"])
    no_amount = {
        "answer": "Цена указывается в рублях, точной суммы в базе нет.",
        "sources": [],
    }
    assert eval_golden.score_record(record, no_amount, k=5)["forbidden"] == []

    with_amount = {"answer": "Абонентская плата 1 500 руб. в месяц.", "sources": []}
    assert eval_golden.score_record(record, with_amount, k=5)["forbidden"] == [
        "1 500 руб"
    ]

    with_name = {"answer": "Расчёты ведёт Иванова.", "sources": []}
    assert eval_golden.score_record(record, with_name, k=5)["forbidden"] == ["Иванова"]
