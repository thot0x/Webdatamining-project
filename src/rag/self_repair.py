"""
self_repair.py — Boucle de génération et auto-correction SPARQL via LLM.
Le LLM génère, valide, et corrige lui-même — pas de template hardcodé.
"""

import os
import json
import time
from typing import Optional, Tuple, List

_HERE = os.path.dirname(__file__)

from sparql_generator import (
    generate_sparql_ollama,
    clean_sparql,
    validate_sparql_syntax,
    validate_sparql_semantics,
    execute_sparql,
    _check_ollama,
    _get_graph,
    _build_query_from_keywords,
    SYSTEM_PROMPT,
    SPARQL_ENDPOINT,
    LLM_MODEL,
    OLLAMA_BASE_URL,
)

MAX_REPAIR_ATTEMPTS = 5   # 5 tentatives pour laisser le LLM corriger

# ── Prompt de réparation ──────────────────────────────────────────────────────

REPAIR_PROMPT_TEMPLATE = """Fix this broken SPARQL query. Output ONLY the corrected query, nothing else.

RULES: Start with PREFIX. FILTER must be INSIDE WHERE before }}. LIMIT after }}.

FIX 1 — FILTER outside WHERE → move it inside:
WRONG:  }} FILTER(?date < "1990-01-01"^^xsd:date) LIMIT 20
RIGHT:  FILTER(?date < "1990-01-01"^^xsd:date) }} LIMIT 20

FIX 2 — Club filter missing → add ex:playsFor with exact URI:
Q "players for Manchester United" → add:  ex:playsFor ex:ManchesterUnited .
Q "players for Barcelona"         → add:  ex:playsFor ex:FcBarcelona .
Q "players for Liverpool"         → add:  ex:playsFor ex:Liverpool .
Q "players for Bayern Munich"     → add:  ex:playsFor ex:BayernMunich .
Q "players for Real Madrid"       → add:  ex:playsFor ex:RealMadrid .
Q "players for Manchester City"   → add:  ex:playsFor ex:ManchesterCity .

FIX 3 — United Kingdom nationality → always use VALUES:
  VALUES ?nat {{ ex:UnitedKingdom ex:UnitedKingdomOfGreatBritainAndIreland }}
  ?player ex:hasNationality ?nat .

PROPERTIES: rdfs:label, ex:birthDate (xsd:date), ex:hasNationality, ex:playsFor, ex:hasPosition, ex:birthPlace, ex:heightM

Error: {error}
Question: {question}

Broken query:
{sparql}

Fixed query:
PREFIX"""


# ── Fonctions de repair ───────────────────────────────────────────────────────

def repair_sparql(question: str, faulty: str, error: str,
                  model: str = LLM_MODEL) -> Optional[str]:
    """
    Envoie la requête cassée + l'erreur au LLM pour correction.
    Retourne la requête nettoyée via clean_sparql, ou None si le LLM échoue.
    """
    if not _check_ollama():
        return None
    try:
        import urllib.request
        payload = json.dumps({
            "model":  model,
            "prompt": REPAIR_PROMPT_TEMPLATE.format(
                error=error, question=question, sparql=faulty
            ),
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 600},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        raw = data.get("response", "").strip()
        return clean_sparql(raw)

    except Exception as e:
        print(f"[REPAIR] Échec appel LLM : {e}")
        return None


def _extract_error(result: Optional[dict], default: str = "") -> str:
    if result is None:
        return default or "Pas de réponse."
    if "error" in result:
        err = result["error"]
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return ""


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate_and_repair(
    question: str,
    endpoint: str = SPARQL_ENDPOINT,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    verbose: bool = True,
) -> Tuple[str, Optional[dict], List[dict]]:
    """
    Pipeline complet NL → SPARQL → validate → execute → repair (LLM).

    1. Le LLM génère le SPARQL initial
    2. On valide la syntaxe via parseQuery
    3. On exécute sur la KB
    4. Si erreur → on renvoie au LLM pour correction (jusqu'à max_attempts fois)
    5. Pas de fallback template — le LLM corrige ou échoue
    """
    repair_log: List[dict] = []

    if not _check_ollama():
        print("[ERROR] Ollama indisponible.")
        return "", None, repair_log

    # Génération initiale — keyword builder first, LLM fallback
    built = _build_query_from_keywords(question)
    if built:
        if verbose:
            print(f"[BUILD] Query assembled from keyword detection (no LLM needed).")
        current_sparql = built
    else:
        if verbose:
            print(f"[GEN  ] Génération SPARQL via LLM ({LLM_MODEL})...")
        raw            = generate_sparql_ollama(question, model=LLM_MODEL)
        current_sparql = clean_sparql(raw)
    current_error  = ""

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"[TRY  ] Tentative {attempt}/{max_attempts}")

        # ── Validation syntaxique ─────────────────────────────────────────────
        ok, msg = validate_sparql_syntax(current_sparql)
        if not ok:
            current_error = msg
            if verbose:
                print(f"[FAIL ] Syntaxe invalide : {msg}")
            repair_log.append({
                "attempt": attempt,
                "sparql":  current_sparql,
                "error":   msg,
                "status":  "syntax_error",
            })
        else:
            # ── Validation sémantique ─────────────────────────────────────────
            ok_sem, msg_sem = validate_sparql_semantics(question, current_sparql)
            if not ok_sem:
                current_error = msg_sem
                if verbose:
                    print(f"[FAIL ] Sémantique invalide : {msg_sem}")
                repair_log.append({
                    "attempt": attempt,
                    "sparql":  current_sparql,
                    "error":   msg_sem,
                    "status":  "semantic_error",
                })
            else:
                # ── Exécution ─────────────────────────────────────────────────
                results = execute_sparql(current_sparql, endpoint)

                if results is not None and "error" not in results:
                    n = len(results.get("results", {}).get("bindings", []))
                    if n == 0 and attempt < max_attempts:
                        current_error = "Query returned 0 results — may be semantically wrong (wrong club/nationality URI, wrong date direction, or wrong entity type)."
                        if verbose:
                            print(f"[WARN ] 0 résultat(s) — possible erreur sémantique, retry...")
                        repair_log.append({
                            "attempt": attempt,
                            "sparql":  current_sparql,
                            "error":   current_error,
                            "status":  "empty_results",
                        })
                    else:
                        repair_log.append({
                            "attempt": attempt,
                            "sparql":  current_sparql,
                            "error":   None,
                            "status":  "success",
                        })
                        if verbose:
                            print(f"[OK   ] Succès en {attempt} tentative(s). {n} résultat(s).")
                        return current_sparql, results, repair_log
                else:
                    current_error = _extract_error(results, "Exécution échouée.")
                    if verbose:
                        print(f"[FAIL ] Erreur exécution : {current_error}")
                    repair_log.append({
                        "attempt": attempt,
                        "sparql":  current_sparql,
                        "error":   current_error,
                        "status":  "execution_error",
                    })

        # ── Repair LLM ────────────────────────────────────────────────────────
        if attempt < max_attempts:
            if verbose:
                print(f"[REPAIR] Envoi au LLM pour correction...")
            repaired = repair_sparql(question, current_sparql, current_error)
            if repaired:
                current_sparql = repaired
                if verbose:
                    print(f"[REPAIR] Requête corrigée reçue.")
            else:
                if verbose:
                    print("[REPAIR] LLM n'a pas pu corriger.")
                break

    if verbose:
        print(f"[FAIL ] Échec après {len(repair_log)} tentative(s).")
    return current_sparql, None, repair_log


# ── Demo ──────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Which football players were born before 1920?",
    "Which players have French nationality?",
    "Which players play for Manchester United?",
    "Which players were born before 1990 and have United Kingdom nationality?",
    "List all football players born after 2000.",
]


def run_demo():
    print("=" * 60)
    print("Self-Repair SPARQL Pipeline Demo")
    print("=" * 60)

    for q in DEMO_QUESTIONS:
        print(f"\n{'─'*60}")
        print(f"[QUESTION] {q}")
        final_sparql, results, log = generate_and_repair(q, verbose=True)

        print(f"[FINAL SPARQL]\n{final_sparql.strip()}")

        if results:
            bindings = results.get("results", {}).get("bindings", [])
            print(f"[RESULTS] {len(bindings)} résultat(s) :")
            for row in bindings[:5]:
                vals = [v["value"].split("/")[-1].replace("_", " ")
                        for v in row.values()]
                print(f"  → {' | '.join(vals)}")
        else:
            print("[RESULTS] Aucun résultat après toutes les tentatives.")

        # Résumé des tentatives
        print(f"[LOG] {len(log)} tentative(s) :")
        for entry in log:
            status = "✓" if entry["status"] == "success" else "✗"
            print(f"  {status} Tentative {entry['attempt']} — {entry['status']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_demo()