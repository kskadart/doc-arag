"""Run the control questions against a running doc-arag API and write a report.

Usage:

    uv run python -m scripts.run_control_questions                 # 12 built-in questions
    uv run python -m scripts.run_control_questions --questions my.json --out reports/run.md

The report records the model configuration reported by `GET /health`, so two
runs with different `LLM_MODEL` / `EMBEDDING_MODEL` values can be compared
side by side. Questions file: JSON list of {"id", "domain", "question",
"expect": "positive" | "no-prices" | "no-names"}.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("control_questions")

DEFAULT_API_URL = "http://localhost:8103"
DEFAULT_OUT_DIR = Path(".claude/reports")
CONFIDENCE_THRESHOLD = 0.7
PRICE_PATTERN = re.compile(r"\d[\d\s]*\s?(руб|₽|р\.)", re.IGNORECASE)

# oreo-data/docs/plan.md §3.2
CONTROL_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q01",
        "domain": "lapa-navigation",
        "expect": "positive",
        "question": "Как войти в Лапу и какие основные разделы в ней есть?",
    },
    {
        "id": "q02",
        "domain": "billing",
        "expect": "positive",
        "question": "Что можно посмотреть по абоненту в биллинге BG Billing?",
    },
    {
        "id": "q03",
        "domain": "diagnostics",
        "expect": "positive",
        "question": "У абонента не работает интернет — какие шаги проверки нужно выполнить?",
    },
    {
        "id": "q04",
        "domain": "diagnostics",
        "expect": "positive",
        "question": "Какие вопросы задать абоненту при обращении по услуге КПД?",
    },
    {
        "id": "q05",
        "domain": "speech-modules",
        "expect": "positive",
        "question": "Как правильно ответить абоненту, который недоволен скоростью интернета?",
    },
    {
        "id": "q06",
        "domain": "products-tariffs",
        "expect": "positive",
        "question": "Какие тарифные модели существуют и чем они отличаются?",
    },
    {
        "id": "q07",
        "domain": "network-equipment",
        "expect": "positive",
        "question": "Чем подключение по PPPoE отличается от статического IP и что такое ELAN?",
    },
    {
        "id": "q08",
        "domain": "connection-process",
        "expect": "positive",
        "question": "Из каких стадий состоит подключение нового клиента и кто отвечает за каждую?",
    },
    {
        "id": "q09",
        "domain": "docflow-acts",
        "expect": "positive",
        "question": "Как оформляется акт сдачи-приёмки выполненных работ?",
    },
    {
        "id": "q10",
        "domain": "sla-tickets",
        "expect": "positive",
        "question": "В какой системе регистрируются обращения клиентов и как отследить заявку?",
    },
    {
        "id": "q11",
        "domain": "products-tariffs",
        "expect": "no-prices",
        "question": "Сколько стоит тариф для юридических лиц?",
    },
    {
        "id": "q12",
        "domain": "connection-process",
        "expect": "no-names",
        "question": "Кто занимается расчётами при подключении?",
    },
]


def load_questions(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return CONTROL_QUESTIONS
    return [dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_blacklist(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_answer(
    item: dict[str, str], result: dict[str, Any], blacklist: list[str]
) -> list[str]:
    """Return a list of failed checks (empty means pass)."""
    failures: list[str] = []
    answer = str(result.get("answer", ""))
    if result.get("sources_used", 0) <= 0:
        failures.append("sources_used=0")
    if float(result.get("confidence", 0.0)) < CONFIDENCE_THRESHOLD:
        failures.append(f"confidence<{CONFIDENCE_THRESHOLD}")
    if not re.search(r"[а-яА-ЯёЁ]", answer):
        failures.append("answer not in Russian")
    if item.get("expect") == "no-prices" and PRICE_PATTERN.search(answer):
        failures.append("answer contains a price")
    if item.get("expect") == "no-names":
        for name in blacklist:
            if name.lower() in answer.lower():
                failures.append(f"answer contains blacklisted '{name}'")
                break
    return failures


def ask(
    client: httpx.Client, api_url: str, question: str, domain: str | None
) -> tuple[dict[str, Any], float]:
    body: dict[str, Any] = {"query": question}
    if domain:
        body["domain"] = domain
    started = time.monotonic()
    response = client.post(f"{api_url}/query", json=body)
    latency = time.monotonic() - started
    response.raise_for_status()
    return dict(response.json()), latency


def render_markdown(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Control questions — {config.get('run_at')}",
        "",
        f"- llm: `{config.get('llm_provider')}` / `{config.get('llm_model')}`",
        f"- embeddings: `{config.get('embedding_model')}`",
        f"- reranker: `{config.get('reranker_provider')}`",
        f"- domain filter: `{config.get('domain_filter')}`",
        "",
        "| id | pass | conf | sources | latency s | question |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        status = "✅" if not row["failures"] else "❌ " + "; ".join(row["failures"])
        lines.append(
            f"| {row['id']} | {status} | {row['confidence']:.2f} | {row['sources_used']} | "
            f"{row['latency']:.1f} | {row['question']} |"
        )
    passed = sum(1 for row in rows if not row["failures"])
    lines += ["", f"**{passed}/{len(rows)} passed**", ""]
    for row in rows:
        lines += [f"## {row['id']} — {row['question']}", "", row["answer"], ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--questions", type=Path, help="JSON file with questions (default: built-in 12)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="markdown report path (default: .claude/reports/<timestamp>.md)",
    )
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=Path("/Users/kskada/develop/oreo-data/corpus/fio-blacklist.txt"),
    )
    parser.add_argument(
        "--use-domain-filter",
        action="store_true",
        help="send each question's domain as a retrieval filter",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions)
    blacklist = load_blacklist(args.blacklist)
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{args.api_url}/health")
        health.raise_for_status()
        config = {
            **health.json(),
            "run_at": run_at,
            "domain_filter": args.use_domain_filter,
        }
        logger.info(
            f"llm={config.get('llm_model')} embeddings={config.get('embedding_model')} "
            f"reranker={config.get('reranker_provider')}"
        )
        for item in questions:
            domain = item.get("domain") if args.use_domain_filter else None
            try:
                result, latency = ask(client, args.api_url, item["question"], domain)
            except httpx.HTTPError as exc:
                logger.error(f"{item['id']}: request failed: {exc}")
                rows.append(
                    {
                        **item,
                        "answer": f"ERROR: {exc}",
                        "confidence": 0.0,
                        "sources_used": 0,
                        "latency": 0.0,
                        "failures": ["request failed"],
                    }
                )
                continue
            failures = check_answer(item, result, blacklist)
            rows.append(
                {
                    **item,
                    "answer": result.get("answer", ""),
                    "confidence": float(result.get("confidence", 0.0)),
                    "sources_used": int(result.get("sources_used", 0)),
                    "latency": latency,
                    "failures": failures,
                }
            )
            logger.info(
                f"{item['id']} {'PASS' if not failures else 'FAIL ' + '; '.join(failures)} "
                f"conf={rows[-1]['confidence']:.2f} sources={rows[-1]['sources_used']} {latency:.1f}s"
            )

    out = args.out or DEFAULT_OUT_DIR / f"control-{run_at}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(config, rows), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(
            json.dumps({"config": config, **row}, ensure_ascii=False) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    passed = sum(1 for row in rows if not row["failures"])
    logger.info(f"{passed}/{len(rows)} passed, report: {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
