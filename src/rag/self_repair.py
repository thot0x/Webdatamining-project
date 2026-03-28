"""
self_repair.py — Auto-correction loop for SPARQL queries.
If a generated query fails, send the error back to the LLM for repair.
"""
 
import os
import re
import json
import time
from typing import Optional, Tuple, List
 
_HERE = os.path.dirname(__file__)
 
from sparql_generator import (
    nl_to_sparql,
    clean_sparql,
    validate_sparql_syntax,
    execute_sparql,
    generate_sparql_ollama,
    _check_ollama,
    SYSTEM_PROMPT,
    SPARQL_ENDPOINT,
    LLM_MODEL,
    OLLAMA_BASE_URL,
)
 
MAX_REPAIR_ATTEMPTS = 3
 
 
#Repair prompt
 
REPAIR_PROMPT_TEMPLATE = """The following SPARQL query produced an error.
Correct the query so it is valid. Return ONLY the corrected SPARQL query.
 
Original question: {question}
 
Faulty SPARQL:
{sparql}
 
Error message:
{error}
 
Corrected SPARQL:"""
 
 
# Error extraction
 
def _extract_error_from_response(result: Optional[dict], raw_exception: str = "") -> str:
    """Return a concise error description from a SPARQL endpoint response."""
    if result is None:
        return raw_exception or "No response from SPARQL endpoint."
    if "error" in result:
        err = result["error"]
        if isinstance(err, dict):
            return err.get("message", str(err))
        return str(err)
    return ""
 
 
# Repair loop
 
def repair_sparql(
    question: str,
    faulty_sparql: str,
    error_msg: str,
    model: str = LLM_MODEL,
) -> Optional[str]:
    """Ask the LLM to fix a broken SPARQL query. Returns corrected query or None."""
    if not _check_ollama():
        return None
 
    prompt = REPAIR_PROMPT_TEMPLATE.format(
        question=question,
        sparql=faulty_sparql,
        error=error_msg,
    )
 
    try:
        import urllib.request
 
        payload = json.dumps({
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 512},
        }).encode()
 
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
 
        raw    = data.get("response", "").strip()
        repaired = clean_sparql(raw)
        return repaired
 
    except Exception as e:
        print(f"[REPAIR] LLM call failed: {e}")
        return None
 
 
#Main pipeline
 
def generate_and_repair(
    question: str,
    endpoint: str = SPARQL_ENDPOINT,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    verbose: bool = True,
) -> Tuple[str, Optional[dict], List[dict]]:
    """
    Full generate → validate → execute → repair loop.
 
    Returns:
        final_sparql  : the best SPARQL string produced
        results       : SPARQL endpoint results dict, or None
        repair_log    : list of attempt dicts with sparql/error/status
    """
    repair_log: List[dict] = []
 
    # Step 1: initial generation
    sparql = nl_to_sparql(question)
    if verbose:
        print(f"[GEN  ] Initial query generated.")
 
    current_sparql = sparql
    current_error  = ""
 
    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"[TRY  ] Attempt {attempt}/{max_attempts}")
 
        # Syntax check
        ok, msg = validate_sparql_syntax(current_sparql)
        if not ok:
            current_error = f"Syntax error: {msg}"
            if verbose:
                print(f"[FAIL ] {current_error}")
            repair_log.append({
                "attempt": attempt,
                "sparql":  current_sparql,
                "error":   current_error,
                "status":  "syntax_error",
            })
        else:
            # Execute
            try:
                results = execute_sparql(current_sparql, endpoint)
            except Exception as exc:
                results = None
                current_error = str(exc)
 
            if results is not None and "error" not in results:
                # Success
                repair_log.append({
                    "attempt": attempt,
                    "sparql":  current_sparql,
                    "error":   None,
                    "status":  "success",
                })
                if verbose:
                    n = len(results.get("results", {}).get("bindings", []))
                    print(f"[OK   ] Query executed successfully. {n} result(s).")
                return current_sparql, results, repair_log
            else:
                current_error = _extract_error_from_response(
                    results, "Endpoint returned no results."
                )
                if verbose:
                    print(f"[FAIL ] Execution error: {current_error}")
                repair_log.append({
                    "attempt": attempt,
                    "sparql":  current_sparql,
                    "error":   current_error,
                    "status":  "execution_error",
                })
 
        # Attempt repair
        if attempt < max_attempts:
            if verbose:
                print(f"[REPAIR] Requesting LLM repair (attempt {attempt})...")
            repaired = repair_sparql(question, current_sparql, current_error)
            if repaired:
                current_sparql = repaired
                if verbose:
                    print(f"[REPAIR] Received repaired query.")
            else:
                if verbose:
                    print("[REPAIR] LLM repair unavailable.")
                break
 
    # All attempts failed — return last attempt
    if verbose:
        print(f"[DONE ] Returning last query after {len(repair_log)} attempt(s).")
    return current_sparql, None, repair_log
 
 
# Demo
 
DEMO_QUESTIONS = [
    "Which football players were born before 1990?",
    "List all clubs in the Premier League.",
    "Who are the French players?",
    "What is the position of Lionel Messi?",
    "Which players have a market value over 50 million euros?",
]
 
 
def run_demo():
    print("=" * 60)
    print("Self-Repair SPARQL Pipeline Demo")
    print("=" * 60)
 
    for q in DEMO_QUESTIONS:
        print(f"\n{'─'*60}")
        print(f"[QUESTION] {q}")
        final_sparql, results, log = generate_and_repair(
            q, endpoint=SPARQL_ENDPOINT, verbose=True
        )
        print(f"[FINAL SPARQL]\n{final_sparql.strip()}")
        if results:
            bindings = results.get("results", {}).get("bindings", [])
            print(f"[RESULTS] {len(bindings)} row(s):")
            for row in bindings[:5]:
                print(f"  {row}")
        print(f"[LOG] {len(log)} attempt(s) made.")
 
    print("\n" + "=" * 60)
 
 
if __name__ == "__main__":
    run_demo()