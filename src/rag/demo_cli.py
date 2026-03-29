"""
demo_cli.py — 5 questions baseline vs RAG pipeline (LLM only, no hardcoded templates)
Usage: python demo_cli.py [--interactive]
"""

import os
import sys
import time
import argparse
import json

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

from sparql_generator import (
    _check_ollama, KB_PATH, LLM_MODEL,
    validate_sparql_syntax, execute_sparql,
)
from self_repair import generate_and_repair, SPARQL_ENDPOINT

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────

USE_COLOUR = sys.platform != "win32"
def c(text, code): return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text
CYAN   = lambda t: c(t, "96")
GREEN  = lambda t: c(t, "92")
YELLOW = lambda t: c(t, "93")
RED    = lambda t: c(t, "91")
BOLD   = lambda t: c(t, "1")
DIM    = lambda t: c(t, "2")

# ── Baseline : réponses statiques (sans KB) ──────────────────────────────────

BASELINE_ANSWERS = {
    "Which football players were born before 1920?":
        "Historical players: Billy Meredith (1874), Steve Bloomer (1874), Bob Crompton (1879), Sam Hardy (1883)...",
    "Which players have French nationality?":
        "French players: Zidane, Platini, Thierry Henry, Mbappe, Griezmann, Cantona, Desailly...",
    "Which players play for Manchester United?":
        "Man United players: Rashford, Bruno Fernandes, Dalot, Onana, Maguire, Mount...",
    "Which players were born before 1990 and have United Kingdom nationality?":
        "UK players born before 1990: Beckham (1975), Gerrard (1980), Lampard (1978), Scholes (1974), Rooney (1985)...",
    "List all football players born after 2000.":
        "Young players: Bellingham (2003), Gavi (2004), Pedri (2002), Camavinga (2002), Yamal (2007)...",
}

DEMO_QUESTIONS = list(BASELINE_ANSWERS.keys())


def baseline_answer(question: str) -> str:
    return BASELINE_ANSWERS.get(question, "No baseline answer available.")


# ── RAG answer ────────────────────────────────────────────────────────────────

def rag_answer(question: str) -> dict:
    t0 = time.time()
    final_sparql, results, repair_log = generate_and_repair(
        question, endpoint=SPARQL_ENDPOINT, verbose=False
    )
    elapsed  = time.time() - t0
    attempts = len(repair_log)
    bindings = results.get("results", {}).get("bindings", []) if results else []
    has_error = results is not None and "error" in results
    success   = results is not None and not has_error

    if success and bindings:
        rows = []
        for b in bindings[:10]:
            vals = [v["value"].split("/")[-1].replace("_", " ") for v in b.values()]
            rows.append(" | ".join(vals))
        answer = f"Trouvé {len(bindings)} résultat(s) :\n  " + "\n  ".join(rows)
    elif success and not bindings:
        answer = "Requête exécutée — aucun résultat dans la KB."
    elif has_error:
        answer = f"Erreur SPARQL : {results['error']['message'][:200]}"
    else:
        answer = f"Échec LLM après {attempts} tentative(s).\nDernier SPARQL :\n{final_sparql[:300]}"

    return {
        "answer":     answer,
        "sparql":     final_sparql,
        "attempts":   attempts,
        "success":    success,
        "n_results":  len(bindings),
        "time_s":     round(elapsed, 2),
        "repair_log": repair_log,
    }


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_header():
    kb_ok = os.path.exists(KB_PATH)
    print(BOLD("═" * 64))
    print(BOLD("  Football Knowledge Graph — RAG CLI Demo"))
    print(BOLD("═" * 64))
    print(f"  KB      : {GREEN('✓ ' + os.path.basename(KB_PATH)) if kb_ok else RED('✗ introuvable')}")
    print(f"  LLM     : {'✓ ' + LLM_MODEL if _check_ollama() else RED('✗ Ollama absent — lance: ollama serve')}")
    print(f"  Moteur  : {GREEN('rdflib local (sans Fuseki)')}")
    print(f"  Baseline: {DIM('LLM seul (sans KB) vs RAG (LLM + KB)')}")
    print(BOLD("═" * 64))


def print_comparison(question: str, baseline: str, rag: dict, idx: int):
    print(f"\n{BOLD(f'Q{idx}.')} {CYAN(question)}")
    print()

    # Baseline LLM sans KB
    print(f"  {YELLOW('▶ BASELINE')} {DIM('(réponse statique — sans KB)')}")
    for line in baseline.strip().splitlines():
        print(f"    {line}")
    print()

    # RAG
    attempts_detail = f" [{rag['attempts']} tentatives LLM]" if rag["attempts"] > 1 else ""
    status = GREEN(f"✓ SUCCESS ({rag['n_results']} résultats KB)") if rag["success"] else RED("✗ ÉCHEC LLM")
    print(f"  {GREEN('▶ RAG PIPELINE')} [{status}]{attempts_detail} [{rag['time_s']}s]")
    for line in rag["answer"].strip().splitlines():
        print(f"    {line}")

    # SPARQL généré
    print(f"\n  {DIM('SPARQL généré par LLM :')}")
    for line in rag["sparql"].strip().splitlines()[:8]:
        print(f"    {DIM(line)}")
    if rag["sparql"].count("\n") > 7:
        print(f"    {DIM('...')}")

    # Détail tentatives
    if rag["attempts"] > 1:
        print(f"\n  {DIM('Historique des tentatives :')}")
        for entry in rag["repair_log"]:
            icon   = "✓" if entry["status"] == "success" else "✗"
            num    = entry["attempt"]
            status = entry["status"]
            line   = f"{icon} Tentative {num} — {status}"
            print(f"    {DIM(line)}")

    print("─" * 64)


# ── Mode interactif ───────────────────────────────────────────────────────────

def interactive_loop():
    print(f"\n{BOLD('Mode interactif')} — pose ta question football, ou tape quit.\n")
    while True:
        try:
            q = input(CYAN("Question : ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break
        if q.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
        if not q:
            continue

        result = rag_answer(q)
        print(f"\n{GREEN('Réponse :')}")
        for line in result["answer"].splitlines():
            print(f"  {line}")
        print(f"\n{DIM('SPARQL généré :')}")
        for line in result["sparql"].strip().splitlines():
            print(f"  {DIM(line)}")
        if result["attempts"] > 1:
            n_att = result["attempts"]
            print(DIM(f"  ({n_att} tentatives LLM)"))
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(interactive: bool = False):
    print_header()

    if interactive:
        interactive_loop()
        return

    print(f"\n{BOLD('5 questions — Baseline (LLM sans KB) vs RAG (LLM + KB)')}\n")

    summary = []
    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        baseline = baseline_answer(question)
        rag      = rag_answer(question)
        print_comparison(question, baseline, rag, i)
        summary.append({
            "question":        question,
            "baseline":        baseline,
            "baseline_source": f"LLM ({LLM_MODEL}) sans KB",
            "rag":             {k: v for k, v in rag.items() if k != "repair_log"},
        })
        time.sleep(0.3)

    # Tableau récap
    print(f"\n{BOLD('═'*64)}")
    print(f"{BOLD('  RÉSUMÉ')}")
    print(f"{BOLD('═'*64)}")
    print(f"  {'#':>2}  {'Question':42s}  {'OK':8s}  {'Essais':6s}  {'Résultats':9s}  {'Temps':6s}")
    print("  " + "─" * 76)
    for i, s in enumerate(summary, 1):
        r  = s["rag"]
        ok = GREEN("✓") if r["success"] else RED("✗")
        print(
            f"  {i:>2}  {s['question'][:40]:42s}  "
            f"{ok:8s}  {r['attempts']:6d}  {r['n_results']:9d}  {r['time_s']:6.2f}s"
        )
    print(f"{BOLD('═'*64)}\n")

    # Sauvegarde
    out = os.path.join(_HERE, "..", "..", "reports", "demo_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[SAVED] Résultats → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football KG RAG Demo CLI")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Mode interactif")
    args = parser.parse_args()
    run(interactive=args.interactive)