"""The System 1 comparison: both deciders over every Labelled set, with accuracy, latency and cost.

Run: uv run python -m agentic_recruiter.evals.system1 [--implementations llm jev] [--workers 4]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from agentic_recruiter.evals.metrics import percentile
from agentic_recruiter.system1 import System1Decider

DECISIONS = {
    "input_guardrail": (lambda d, ex: d.input_guardrail(ex["text"]), lambda r: r.verdict),
    "route": (lambda d, ex: d.route(ex["question"]), lambda r: r.collection),
    "chunk_relevance": (lambda d, ex: d.chunk_relevance(ex["question"], [ex["chunk"]]), lambda r: r.keep[0]),
    "groundedness": (lambda d, ex: d.groundedness(ex["claim"], ex["evidence"]), lambda r: r.grounded),
    "turn_check": (lambda d, ex: d.turn_check(ex["question"], ex["answer"]), lambda r: r.label),
}


def run_system1_comparison(
    deciders: dict[str, System1Decider], data_dir: Path, out_dir: Path, *, workers: int = 4
) -> dict:
    records = []
    for name, decider in deciders.items():
        for decision, (call, read) in DECISIONS.items():
            path = Path(data_dir) / f"{decision}.jsonl"
            if not path.exists():
                continue
            examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            with ThreadPoolExecutor(workers) as pool:
                records += pool.map(lambda ex: _record(name, decision, decider, call, read, ex), examples)

    summary = _summarise(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(_markdown(summary, workers))
    return summary


def _record(implementation, decision, decider, call, read, example) -> dict:
    record = {"implementation": implementation, "decision": decision, "id": example["id"], "expected": example["label"]}
    try:
        result = call(decider, example)
        predicted = read(result)
        return {**record, "predicted": predicted, "correct": predicted == example["label"],
                "latency_s": result.trace.latency_s, "cost_usd": result.trace.cost_usd, "error": None}
    except Exception as error:  # a failed decision counts as wrong and is reported
        return {**record, "predicted": None, "correct": False, "latency_s": None, "cost_usd": 0.0,
                "error": f"{type(error).__name__}: {error}"}


def _summarise(records: list[dict]) -> dict:
    summary: dict = {}
    for r in records:
        summary.setdefault(r["implementation"], {}).setdefault(r["decision"], []).append(r)
    for implementation, by_decision in summary.items():
        for decision, rows in by_decision.items():
            latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
            stats = {
                "n": len(rows),
                "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "errors": sum(r["error"] is not None for r in rows),
                "latency_p50_s": percentile(latencies, 50),
                "latency_p95_s": percentile(latencies, 95),
                "cost_total_usd": sum(r["cost_usd"] for r in rows),
                "cost_mean_usd": sum(r["cost_usd"] for r in rows) / len(rows),
            }
            if decision == "input_guardrail":
                stats["missed_discriminatory"] = sum(r["expected"] == "discriminatory" and r["predicted"] == "allowed" for r in rows)
                stats["wrongly_refused"] = sum(r["expected"] == "allowed" and r["predicted"] not in ("allowed", None) for r in rows)
            if decision == "turn_check":
                answers = ("substantive", "shallow")
                stats["missed_injection"] = sum(r["expected"] == "injection" and r["predicted"] != "injection" for r in rows)
                stats["interrupted_answers"] = sum(
                    r["expected"] in answers and r["predicted"] not in (*answers, None) for r in rows
                )
            by_decision[decision] = stats
    return summary


def _markdown(summary: dict, workers: int) -> str:
    lines = [
        "# System 1 comparison", "",
        f"Run {datetime.now().isoformat(timespec='seconds')}, {workers} concurrent requests per implementation. "
        "Latency is end-to-end through OpenRouter.", "",
        "| decision | implementation | n | accuracy | p50 latency (s) | p95 latency (s) | mean cost ($) | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    decisions = sorted({d for by in summary.values() for d in by})
    for decision in decisions:
        for implementation in summary:
            s = summary[implementation].get(decision)
            if s:
                lines.append(
                    f"| {decision} | {implementation} | {s['n']} | {s['accuracy']:.0%} | {s['latency_p50_s']:.3f} | "
                    f"{s['latency_p95_s']:.3f} | {s['cost_mean_usd']:.7f} | {s['errors']} |"
                )
    if {"llm", "jev"} <= set(summary):
        lines += ["", "| decision | Jev speed-up (p50) | Jev cost ratio (mean) | accuracy gap (Jev − LLM) |", "|---|---|---|---|"]
        for decision in decisions:
            llm, jev = summary["llm"].get(decision), summary["jev"].get(decision)
            if llm and jev and jev["latency_p50_s"] and jev["cost_mean_usd"]:
                lines.append(
                    f"| {decision} | {llm['latency_p50_s'] / jev['latency_p50_s']:.1f}× | "
                    f"{llm['cost_mean_usd'] / jev['cost_mean_usd']:.0f}× cheaper | "
                    f"{(jev['accuracy'] - llm['accuracy']) * 100:+.0f} pts |"
                )
    guard = {i: s.get("input_guardrail") for i, s in summary.items() if s.get("input_guardrail")}
    if guard:
        lines += ["", "Input guardrail errors that matter most:"]
        lines += [f"- {i}: {g['missed_discriminatory']} discriminatory requests allowed, {g['wrongly_refused']} legitimate requests refused" for i, g in guard.items()]
    turn = {i: s.get("turn_check") for i, s in summary.items() if s.get("turn_check")}
    if turn:
        lines += ["", "Turn check errors that matter most:"]
        lines += [
            f"- {i}: {t['missed_injection']} prompt injections missed, {t['interrupted_answers']} real answers reprompted or redirected"
            for i, t in turn.items()
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementations", nargs="+", default=["llm", "jev"])
    parser.add_argument("--data", type=Path, default=Path("evals/data"))
    parser.add_argument("--out", type=Path, default=Path("results") / f"system1-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from agentic_recruiter.config import load_config
    from agentic_recruiter.openrouter import OpenRouterGateway
    from agentic_recruiter.system1 import make_deciders

    deciders = make_deciders(load_config(), OpenRouterGateway.from_env())
    run_system1_comparison({i: deciders[i] for i in args.implementations}, args.data, args.out, workers=args.workers)
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
