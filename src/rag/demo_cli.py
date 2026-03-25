"""
demo_cli.py — Interactive CLI: 5 baseline questions vs RAG pipeline
Usage: python demo_cli.py [--interactive]
"""
 
import os
import sys
import time
import argparse
import json
from typing import Optional
 
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
 
from sparql_generator import nl_to_sparql, execute_sparql, _check_ollama, SPARQL_ENDPOINT
from self_repair import generate_and_repair
 
#ANSI colours (disabled on Windows)
 
USE_COLOUR = sys.platform != "win32"
 
def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text
 
CYAN   = lambda t: c(t, "96")
GREEN  = lambda t: c(t, "92")
YELLOW = lambda t: c(t, "93")
RED    = lambda t: c(t, "91")
BOLD   = lambda t: c(t, "1")
DIM    = lambda t: c(t, "2")
 
 
#Baseline (no RAG): keyword-heuristic answers
 
BASELINE_ANSWERS = {
    "Which players play for Real Madrid?":
        "Based on general football knowledge: Bellingham, Vinicius Jr, Mbappe, Modric, Kroos.",
    "Who are the French players in the knowledge graph?":
        "Common French players: Mbappe, Griezmann, Giroud, Pogba.",
    "List all clubs that compete in the Premier League.":
        "Premier League clubs include: Man City, Arsenal, Liverpool, Chelsea, Man United, Tottenham...",
    "Which players were born before 1990?":
        "Players born before 1990 include: Messi (1987), Ronaldo (1985), Modric (1985).",
    "What position does Kylian Mbappe play?":
        "Kylian Mbappe plays as a Forward / Striker.",
}
 
DEMO_QUESTIONS = list(BASELINE_ANSWERS.keys())
 
 
#RAG answer
 
def rag_answer(question: str, endpoint: str = SPARQL_ENDPOINT) -> dict:
    """
    Full RAG pipeline:
      1. Generate SPARQL from NL
      2. Execute on KB (with self-repair)
      3. Format results as a human answer
    """
    t0 = time.time()
 
    final_sparql, results, repair_log = generate_and_repair(
        question, endpoint=endpoint, verbose=False
    )
 
    elapsed = time.time() - t0
    attempts = len(repair_log)
    success  = results is not None and "error" not in (results or {})
 
    # Format answer
    if success and results:
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            # Extract text from first variable of each binding
            rows = []
            for b in bindings[:10]:
                vals = [v.get("value", "") for v in b.values()]
                rows.append(" | ".join(vals))
            answer = "\n  ".join(rows)
            answer = f"Found {len(bindings)} result(s):\n  {answer}"
        else:
            answer = "Query executed but returned no results."
    else:
        answer = f"Could not retrieve results from knowledge graph. Last SPARQL:\n{final_sparql[:300]}"
 
    return {
        "answer":   answer,
        "sparql":   final_sparql,
        "attempts": attempts,
        "success":  success,
        "time_s":   round(elapsed, 2),
        "repair_log": repair_log,
    }
 
 
#Display helpers
 
def print_header():
    print(BOLD("═" * 64))
    print(BOLD("  Football Knowledge Graph — RAG CLI Demo"))
    print(BOLD("═" * 64))
    print(f"  Endpoint : {DIM(SPARQL_ENDPOINT)}")
    print(f"  Ollama   : {'✓ available' if _check_ollama() else RED('✗ not reachable (template fallback)')}")
    print(BOLD("═" * 64))
 
 
def print_comparison(question: str, baseline: str, rag: dict, idx: int):
    print(f"\n{BOLD(f'Q{idx}.')} {CYAN(question)}")
    print()
 
    # Baseline
    print(f"  {YELLOW('▶ BASELINE')} {DIM('(heuristic / static)')}")
    for line in baseline.strip().splitlines():
        print(f"    {line}")
 
    print()
    # RAG
    status = GREEN("✓ SUCCESS") if rag["success"] else RED("✗ FAILED")
    print(
        f"  {GREEN('▶ RAG PIPELINE')} "
        f"[{status}] "
        f"[{rag['attempts']} attempt(s)] "
        f"[{rag['time_s']}s]"
    )
    for line in rag["answer"].strip().splitlines():
        print(f"    {line}")
 
    # SPARQL
    print(f"\n  {DIM('Generated SPARQL:')}")
    for line in rag["sparql"].strip().splitlines()[:10]:
        print(f"    {DIM(line)}")
    if rag["sparql"].count("\n") > 9:
        print(f"    {DIM('...')}")
    print("─" * 64)
 
 
# Interactive mode
 
def interactive_loop():
    print(f"\n{BOLD('Interactive mode')} — type a football question, or 'quit' to exit.\n")
    while True:
        try:
            q = input(CYAN("Your question: ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not q:
            continue
 
        result = rag_answer(q)
        print(f"\n{GREEN('Answer:')}")
        for line in result["answer"].splitlines():
            print(f"  {line}")
        print(f"\n{DIM('SPARQL:')}")
        for line in result["sparql"].strip().splitlines():
            print(f"  {DIM(line)}")
        print()
 
 
#Main
 
def run(interactive: bool = False):
    print_header()
 
    if interactive:
        interactive_loop()
        return
 
    # Run the 5 demo questions
    print(f"\n{BOLD('Running 5 demo questions: Baseline vs RAG')}\n")
 
    summary = []
    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        baseline = BASELINE_ANSWERS[question]
        rag      = rag_answer(question)
        print_comparison(question, baseline, rag, i)
        summary.append({
            "question":  question,
            "baseline":  baseline,
            "rag":       {k: v for k, v in rag.items() if k != "repair_log"},
        })
        time.sleep(0.5)   # Be gentle on Ollama
 
    # Summary table
    print(f"\n{BOLD('═'*64)}")
    print(f"{BOLD('  SUMMARY TABLE')}")
    print(f"{BOLD('═'*64)}")
    print(f"  {'#':>2}  {'Question':50s}  {'Success':7s}  {'Attempts':8s}  {'Time':6s}")
    print("  " + "─" * 80)
    for i, s in enumerate(summary, 1):
        r   = s["rag"]
        ok  = "✓" if r["success"] else "✗"
        print(
            f"  {i:>2}  {s['question'][:48]:50s}  "
            f"{ok:7s}  {r['attempts']:8d}  {r['time_s']:6.2f}s"
        )
    print(f"{BOLD('═'*64)}\n")
 
    # Save results
    out = os.path.join(_HERE, "..", "..", "reports", "demo_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[SAVED] Demo results → {out}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football KG RAG Demo CLI")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive question loop")
    args = parser.parse_args()
    run(interactive=args.interactive)