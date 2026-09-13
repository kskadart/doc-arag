"""Validate the golden set against the corpus and evaluate a running doc-arag on it.

Usage:

    uv run python -m scripts.eval_golden --validate            # merge parts/*.jsonl -> golden-set.jsonl
    uv run python -m scripts.eval_golden                       # run every record against the API
    uv run python -m scripts.eval_golden --domain diagnostics --judge --judge-model qwen/qwen3.7-plus

Metrics per record: doc_hit (source file among returned sources), domain_hit,
fact_coverage (share of `must_include` found in the answer), forbidden
(`must_not_include` hit) and optional judge score 1-5 from an LLM comparing
the answer with `reference_answer`. The report records the model configuration
from `GET /health`, so runs with different settings can be compared.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("eval_golden")

DEFAULT_GOLDEN_DIR = Path("/Users/kskada/develop/oreo-data/golden")
DEFAULT_CORPUS = Path("/Users/kskada/develop/oreo-data/corpus")
DEFAULT_API_URL = "http://localhost:8103"
DEFAULT_OUT_DIR = Path(".claude/reports")
RECORD_TYPES = {"factual", "procedural", "paraphrase", "cross-doc", "negative"}
# A forbidden currency word counts only next to a number: "в рублях" is not a leak
CURRENCY_TOKENS = ("руб", "рубл", "₽", "р.")
PRICE_PATTERN = re.compile(r"\d[\d\s.,]*\s?(руб|₽|р\.)", re.IGNORECASE)
REQUIRED_FIELDS = (
    "id",
    "domain",
    "source",
    "section",
    "type",
    "question",
    "reference_answer",
    "must_include",
    "must_not_include",
)
JUDGE_PROMPT = """You are grading an answer from a support assistant for a Russian ISP.
Compare the ANSWER with the REFERENCE, which is the ground truth from the knowledge base.

Question: {question}

REFERENCE:
{reference}

ANSWER:
{answer}

Score the ANSWER from 1 to 5:
5 - fully correct and complete with respect to the REFERENCE, no contradictions
4 - correct, minor omissions
3 - partially correct or missing an important part
2 - mostly wrong or off-topic, with a fragment of truth
1 - wrong, contradicts the REFERENCE, or refuses without reason

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


# --- golden set loading and validation ----------------------------------------


def load_records(golden_dir: Path) -> list[dict[str, Any]]:
    """Read `golden-set.jsonl` if present, otherwise merge `parts/*.jsonl`."""
    merged = golden_dir / "golden-set.jsonl"
    paths = (
        [merged] if merged.is_file() else sorted((golden_dir / "parts").glob("*.jsonl"))
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            record["_origin"] = f"{path.name}:{line_no}"
            records.append(record)
    return records


def load_parts(golden_dir: Path) -> list[dict[str, Any]]:
    """Merge only `parts/*.jsonl`, ignoring an existing merged file."""
    records: list[dict[str, Any]] = []
    for path in sorted((golden_dir / "parts").glob("*.jsonl")):
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                record = json.loads(line)
                record["_origin"] = f"{path.name}:{line_no}"
                records.append(record)
    return records


def _headings(text: str) -> set[str]:
    return {
        line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")
    }


def validate_records(records: list[dict[str, Any]], corpus_root: Path) -> list[str]:
    """Return every problem found; an empty list means the set is consistent with the corpus."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    cache: dict[str, tuple[str, set[str]]] = {}

    for record in records:
        origin = record.get("_origin", "?")
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            problems.append(f"{origin}: missing fields {missing}")
            continue
        rid = str(record["id"])
        if rid in seen_ids:
            problems.append(f"{origin}: duplicate id {rid}")
        seen_ids.add(rid)
        if record["type"] not in RECORD_TYPES:
            problems.append(f"{origin} ({rid}): unknown type {record['type']!r}")
        if not isinstance(record["must_include"], list) or not record["must_include"]:
            problems.append(f"{origin} ({rid}): must_include must be a non-empty list")
            continue
        if not isinstance(record["must_not_include"], list):
            problems.append(f"{origin} ({rid}): must_not_include must be a list")

        source = corpus_root / str(record["source"])
        if not source.is_file():
            problems.append(f"{origin} ({rid}): source not found: {record['source']}")
            continue
        if source.parent.name != record["domain"]:
            problems.append(
                f"{origin} ({rid}): domain {record['domain']!r} != directory {source.parent.name!r}"
            )
        if str(record["source"]) not in cache:
            text = source.read_text(encoding="utf-8")
            cache[str(record["source"])] = (text.lower(), _headings(text))
        lowered, headings = cache[str(record["source"])]
        if record["section"] and record["section"] not in headings:
            problems.append(
                f"{origin} ({rid}): section not a heading in source: {record['section']!r}"
            )
        for fact in record["must_include"]:
            if str(fact).lower() not in lowered:
                problems.append(
                    f"{origin} ({rid}): must_include {fact!r} not found in {record['source']}"
                )
        if not re.search(r"[а-яё]", str(record["question"]).lower()):
            problems.append(f"{origin} ({rid}): question is not in Russian")
    return problems


def write_merged(records: list[dict[str, Any]], golden_dir: Path) -> Path:
    out = golden_dir / "golden-set.jsonl"
    lines = [
        json.dumps(
            {k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False
        )
        for r in records
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --- scoring ---------------------------------------------------------------------


def score_record(
    record: dict[str, Any], result: dict[str, Any], k: int
) -> dict[str, Any]:
    answer = str(result.get("answer", ""))
    lowered = answer.lower()
    sources = list(result.get("sources", []))[:k]
    source_name = Path(str(record["source"])).name

    found = [f for f in record["must_include"] if str(f).lower() in lowered]
    forbidden: list[str] = []
    tokens = [str(t) for t in record.get("must_not_include", [])]
    if any(t.lower() in CURRENCY_TOKENS for t in tokens):
        price = PRICE_PATTERN.search(answer)
        if price:
            forbidden.append(" ".join(price.group(0).split()))
    forbidden += [
        t for t in tokens if t.lower() not in CURRENCY_TOKENS and t.lower() in lowered
    ]
    return {
        "doc_hit": any(s.get("document_name") == source_name for s in sources),
        "domain_hit": any(s.get("domain") == record["domain"] for s in sources),
        "fact_coverage": len(found) / len(record["must_include"]),
        "facts_missing": [f for f in record["must_include"] if f not in found],
        "forbidden": forbidden,
        "confidence": float(result.get("confidence", 0.0)),
        "sources_used": int(result.get("sources_used", 0)),
        "russian": bool(re.search(r"[а-яё]", lowered)),
    }


def parse_judge_reply(text: str) -> tuple[int | None, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            score = int(payload.get("score"))
            if 1 <= score <= 5:
                return score, str(payload.get("reason", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    digit = re.search(r"\b([1-5])\b", text)
    return (
        (int(digit.group(1)), text.strip()[:200])
        if digit
        else (None, text.strip()[:200])
    )


def judge_answer(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    record: dict[str, Any],
    answer: str,
    extra_body: dict[str, Any] | None = None,
) -> tuple[int | None, str]:
    prompt = JUDGE_PROMPT.format(
        question=record["question"], reference=record["reference_answer"], answer=answer
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if extra_body:
        body.update(extra_body)
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_judge_reply(str(content))


# --- report -------------------------------------------------------------------------


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    judged = [r["judge_score"] for r in rows if r.get("judge_score") is not None]
    return {
        "n": len(rows),
        "doc_hit": sum(r["doc_hit"] for r in rows) / len(rows),
        "domain_hit": sum(r["domain_hit"] for r in rows) / len(rows),
        "fact_coverage": statistics.mean(r["fact_coverage"] for r in rows),
        "forbidden": sum(1 for r in rows if r["forbidden"]),
        "confidence": statistics.mean(r["confidence"] for r in rows),
        "judge": statistics.mean(judged) if judged else None,
    }


def _fmt(stats: dict[str, Any]) -> str:
    judge = f"{stats['judge']:.2f}" if stats.get("judge") is not None else "—"
    return (
        f"| {stats['n']} | {stats['doc_hit']:.0%} | {stats['domain_hit']:.0%} | "
        f"{stats['fact_coverage']:.0%} | {stats['forbidden']} | {stats['confidence']:.2f} | {judge} |"
    )


def render_markdown(config: dict[str, Any], rows: list[dict[str, Any]], k: int) -> str:
    header = "| group | n | doc_hit@k | domain_hit@k | facts | forbidden | conf | judge |\n|---|---|---|---|---|---|---|---|"
    lines = [
        f"# Golden set — {config.get('run_at')}",
        "",
        f"- llm: `{config.get('llm_provider')}` / `{config.get('llm_model')}`",
        f"- embeddings: `{config.get('embedding_model')}`",
        f"- reranker: `{config.get('reranker_provider')}`",
        f"- domain filter: `{config.get('domain_filter')}`, k={k}, judge: `{config.get('judge_model') or 'off'}`",
        "",
        header,
        f"| **all** {_fmt(aggregate(rows))}",
    ]
    for key in ("domain", "type"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        for name, group in sorted(groups.items()):
            lines.append(f"| {key}={name} {_fmt(aggregate(group))}")

    lines += [
        "",
        "## Records",
        "",
        "| id | type | doc | dom | facts | forb | conf | judge | missing facts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        judge = row["judge_score"] if row.get("judge_score") is not None else "—"
        lines.append(
            f"| {row['id']} | {row['type']} | {'✅' if row['doc_hit'] else '❌'} | "
            f"{'✅' if row['domain_hit'] else '❌'} | {row['fact_coverage']:.0%} | "
            f"{'❌ ' + ', '.join(row['forbidden']) if row['forbidden'] else ''} | "
            f"{row['confidence']:.2f} | {judge} | {', '.join(row['facts_missing'])} |"
        )
    lines += ["", "## Answers", ""]
    for row in rows:
        lines += [
            f"### {row['id']} — {row['question']}",
            "",
            f"*Эталон:* {row['reference_answer']}",
            "",
            f"*Ответ:* {row['answer']}",
            "",
        ]
        if row.get("judge_reason"):
            lines += [f"*Judge:* {row['judge_reason']}", ""]
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate parts/*.jsonl against the corpus and write golden-set.jsonl",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--out",
        type=Path,
        help="markdown report path (default: .claude/reports/golden-<timestamp>.md)",
    )
    parser.add_argument(
        "--domain", action="append", help="only records of this domain (repeatable)"
    )
    parser.add_argument("--ids", help="comma-separated record ids")
    parser.add_argument("--limit", type=int, help="stop after N records")
    parser.add_argument(
        "--k", type=int, default=5, help="sources considered for doc/domain hit"
    )
    parser.add_argument(
        "--use-domain-filter",
        action="store_true",
        help="send the record's domain as a retrieval filter",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="grade answers with an LLM against reference_answer",
    )
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("JUDGE_BASE_URL") or os.environ.get("LLM_BASE_URL"),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("JUDGE_API_KEY") or os.environ.get("LLM_API_KEY"),
    )
    parser.add_argument(
        "--judge-extra-body",
        default=os.environ.get("JUDGE_EXTRA_BODY"),
        help='JSON merged into the judge request, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'',
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _load_dotenv()
    args = build_parser().parse_args(argv)

    if args.validate:
        records = load_parts(args.golden_dir)
        problems = validate_records(records, args.corpus)
        for problem in problems:
            logger.error(problem)
        if problems:
            logger.error(
                f"{len(problems)} problem(s) in {len(records)} records, golden-set.jsonl not written"
            )
            return 1
        out = write_merged(records, args.golden_dir)
        by_type: dict[str, int] = defaultdict(int)
        by_domain: dict[str, int] = defaultdict(int)
        for r in records:
            by_type[r["type"]] += 1
            by_domain[r["domain"]] += 1
        logger.info(f"{len(records)} records valid -> {out}")
        logger.info(f"by domain: {dict(sorted(by_domain.items()))}")
        logger.info(f"by type: {dict(sorted(by_type.items()))}")
        return 0

    records = load_records(args.golden_dir)
    problems = validate_records(records, args.corpus)
    if problems:
        for problem in problems:
            logger.error(problem)
        return 1
    if args.domain:
        records = [r for r in records if r["domain"] in set(args.domain)]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        records = [r for r in records if r["id"] in wanted]
    if args.limit:
        records = records[: args.limit]
    if not records:
        logger.error("no records selected")
        return 1
    if args.judge and not (
        args.judge_model and args.judge_base_url and args.judge_api_key
    ):
        logger.error(
            "--judge needs --judge-model and a base url / api key (or LLM_BASE_URL / LLM_API_KEY in env)"
        )
        return 1
    judge_extra_body: dict[str, Any] | None = None
    if args.judge_extra_body:
        try:
            judge_extra_body = json.loads(args.judge_extra_body)
        except json.JSONDecodeError as exc:
            logger.error(f"--judge-extra-body is not valid JSON: {exc}")
            return 1

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{args.api_url}/health")
        health.raise_for_status()
        config = {
            **health.json(),
            "run_at": run_at,
            "domain_filter": args.use_domain_filter,
            "judge_model": args.judge_model if args.judge else None,
        }
        logger.info(
            f"llm={config.get('llm_model')} embeddings={config.get('embedding_model')} "
            f"reranker={config.get('reranker_provider')} records={len(records)}"
        )
        for record in records:
            body: dict[str, Any] = {"query": record["question"]}
            if args.use_domain_filter:
                body["domain"] = record["domain"]
            started = time.monotonic()
            try:
                response = client.post(f"{args.api_url}/query", json=body)
                response.raise_for_status()
                result = dict(response.json())
            except httpx.HTTPError as exc:
                logger.error(f"{record['id']}: request failed: {exc}")
                result = {
                    "answer": f"ERROR: {exc}",
                    "confidence": 0.0,
                    "sources_used": 0,
                    "sources": [],
                }
            latency = time.monotonic() - started

            scores = score_record(record, result, args.k)
            judge_score, judge_reason = None, ""
            if args.judge:
                try:
                    judge_score, judge_reason = judge_answer(
                        client,
                        args.judge_base_url,
                        args.judge_api_key,
                        args.judge_model,
                        record,
                        str(result.get("answer", "")),
                        extra_body=judge_extra_body,
                    )
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    judge_reason = f"judge failed: {exc}"
            row = {
                **{k: v for k, v in record.items() if not k.startswith("_")},
                **scores,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", [])[: args.k],
                "latency": latency,
                "judge_score": judge_score,
                "judge_reason": judge_reason,
            }
            rows.append(row)
            logger.info(
                f"{record['id']} doc={'Y' if row['doc_hit'] else 'n'} dom={'Y' if row['domain_hit'] else 'n'} "
                f"facts={row['fact_coverage']:.0%}{' FORBIDDEN' if row['forbidden'] else ''} "
                f"conf={row['confidence']:.2f}{f' judge={judge_score}' if judge_score else ''} {latency:.1f}s"
            )

    out = args.out or DEFAULT_OUT_DIR / f"golden-{run_at}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(config, rows, args.k), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(
            json.dumps({"config": config, **row}, ensure_ascii=False) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    summary = aggregate(rows)
    logger.info(
        f"doc_hit={summary['doc_hit']:.0%} domain_hit={summary['domain_hit']:.0%} "
        f"facts={summary['fact_coverage']:.0%} forbidden={summary['forbidden']} "
        f"judge={summary['judge'] if summary['judge'] is not None else '—'} report: {out}"
    )
    return 1 if summary["forbidden"] else 0


if __name__ == "__main__":
    sys.exit(main())
