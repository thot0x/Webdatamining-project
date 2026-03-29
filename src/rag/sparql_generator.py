"""
sparql_generator.py — NL → SPARQL via Ollama (gemma:2b or qwen2:0.5b)
Exécution SPARQL locale via rdflib (pas besoin de Fuseki).
Pas de fallback template — le LLM doit générer et corriger lui-même.
"""

import os
import re
import json
from typing import Optional, Tuple

_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.environ.get("LLM_MODEL", "gemma:2b")

_KB_TTL = os.path.join(PROJECT, "kg_artifacts", "expanded_kb.ttl")
_KB_NT  = os.path.join(PROJECT, "kg_artifacts", "expanded_kb.nt")
KB_PATH = _KB_TTL if os.path.exists(_KB_TTL) else _KB_NT

SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT", "local://rdflib")

FOOTBALL_SCHEMA_SUMMARY = """
Prefixes:
  ex:   <http://example.org/>
  rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  xsd:  <http://www.w3.org/2001/XMLSchema#>

Classes: ex:FootballPlayer, ex:Club, ex:Country, ex:Place

Properties (ONLY these exist in the KB):
  rdfs:label         → string  (player name, e.g. "Lionel Messi")
  ex:birthDate       → xsd:date (e.g. "1985-02-05"^^xsd:date)
  ex:birthPlace      → ex:Place URI
  ex:hasNationality  → ex:Country URI (e.g. ex:France, ex:UnitedKingdom)
  ex:playsFor        → ex:Club URI (e.g. ex:ManchesterUnited, ex:RealMadrid)
"""

FEW_SHOT_EXAMPLES = """
## Example 1
Question: Which players were born before 1990?
SPARQL:
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:birthDate ?date .
  FILTER(?date < "1990-01-01"^^xsd:date)
} ORDER BY ?date LIMIT 20

## Example 2
Question: Which players have French nationality?
SPARQL:
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?player ?label WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:hasNationality ex:France .
} LIMIT 20

## Example 3
Question: Which players play for Manchester United?
SPARQL:
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?player ?label WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:playsFor ex:ManchesterUnited .
} LIMIT 20

## Example 4
Question: Which players were born before 1990 and have United Kingdom nationality?
SPARQL:
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:birthDate ?date ;
          ex:hasNationality ex:UnitedKingdom .
  FILTER(?date < "1990-01-01"^^xsd:date)
} ORDER BY ?date LIMIT 20

## Example 5
Question: List all football players born after 2000.
SPARQL:
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:birthDate ?date .
  FILTER(?date > "2000-01-01"^^xsd:date)
} ORDER BY DESC(?date) LIMIT 20
"""

SYSTEM_PROMPT = f"""You are a SPARQL expert. Generate SPARQL queries for a football knowledge graph.

STRICT RULES — follow exactly:
1. Output ONLY the SPARQL query. Zero explanation, zero commentary, zero markdown.
2. Start directly with PREFIX declarations.
3. Use ONLY these prefixes: ex: rdfs: xsd:
4. Always use turtle-style triples:
   ?player a ex:FootballPlayer ;
           rdfs:label ?label ;
           ex:birthDate ?date .
5. FILTER inside WHERE, before the closing brace.
6. LIMIT must be OUTSIDE and AFTER the closing brace }}
7. NO semicolons after LIMIT. NO dot after LIMIT.
8. Country names CamelCase no spaces: ex:France ex:UnitedKingdom ex:Spain
9. Club names CamelCase no spaces: ex:ManchesterUnited ex:RealMadrid

Schema:
{FOOTBALL_SCHEMA_SUMMARY}

Study these examples carefully and follow the exact same pattern:
{FEW_SHOT_EXAMPLES}
"""


# ── rdflib graph singleton ────────────────────────────────────────────────────

_RDF_GRAPH = None

def _get_graph():
    global _RDF_GRAPH
    if _RDF_GRAPH is not None:
        return _RDF_GRAPH
    try:
        from rdflib import Graph
    except ImportError:
        print("[ERROR] rdflib non installé. pip install rdflib")
        return None
    if not os.path.exists(KB_PATH):
        print(f"[WARN] KB introuvable : {KB_PATH}")
        return None
    ext = os.path.splitext(KB_PATH)[1].lower()
    fmt = "turtle" if ext in (".ttl", ".turtle") else "nt"
    print(f"[KB] Chargement {KB_PATH} (format={fmt})...")
    g = Graph()
    g.parse(KB_PATH, format=fmt)
    print(f"[KB] {len(g)} triplets chargés en mémoire.")
    _RDF_GRAPH = g
    return g


# ── Ollama ────────────────────────────────────────────────────────────────────

def _check_ollama():
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def generate_sparql_ollama(question: str, model: str = LLM_MODEL) -> str:
    import urllib.request
    payload = json.dumps({
        "model":  model,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nSPARQL:",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


# ── Post-processing ───────────────────────────────────────────────────────────

def clean_sparql(raw: str) -> str:
    """
    Nettoie et corrige les erreurs communes de gemma:2b :
    - Supprime markdown
    - Supprime texte avant PREFIX/SELECT
    - Corrige PREFIXes: → PREFIX
    - Déplace LIMIT hors du WHERE si mal placé
    - Supprime ; après LIMIT
    """
    # Supprimer markdown
    raw = re.sub(r"```(?:sparql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")

    # Corriger PREFIXes: → PREFIX
    raw = re.sub(r"PREFIXes?\s*:", "PREFIX", raw, flags=re.IGNORECASE)

    # Supprimer tout texte avant PREFIX ou SELECT
    match = re.search(r"(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)", raw, re.IGNORECASE)
    if match:
        raw = raw[match.start():]

    # Supprimer ; après LIMIT N
    raw = re.sub(r"(LIMIT\s+\d+)\s*;", r"\1", raw)

    # Supprimer . après LIMIT N
    raw = re.sub(r"(LIMIT\s+\d+)\s*\.", r"\1", raw)

    # Déplacer LIMIT hors du WHERE s'il est à l'intérieur
    inner_limit = re.search(r"(LIMIT\s+\d+)\s*\n(\s*}\s*)$", raw, re.MULTILINE)
    if inner_limit:
        limit_str   = inner_limit.group(1)
        close_brace = inner_limit.group(2)
        raw = raw[:inner_limit.start()] + close_brace.rstrip() + "\n" + limit_str

    return raw.strip()


def validate_sparql_syntax(sparql: str) -> Tuple[bool, str]:
    """Validation réelle via le parseur SPARQL de rdflib."""
    lower = sparql.lower().strip()
    if not lower.startswith(("select", "prefix", "ask", "construct", "describe")):
        return False, "Ne commence pas par un mot-clé SPARQL valide."
    if "select" not in lower and "ask" not in lower:
        return False, "Pas de clause SELECT."
    if "where" not in lower:
        return False, "Pas de clause WHERE."
    # Validation syntaxique réelle
    g = _get_graph()
    if g is not None:
        try:
            from rdflib.plugins.sparql.parser import parseQuery
            parseQuery(sparql)
        except Exception as e:
            return False, f"Syntaxe invalide : {e}"
    return True, "OK"


# ── Exécution locale via rdflib ───────────────────────────────────────────────

def execute_sparql(sparql: str, endpoint: str = SPARQL_ENDPOINT) -> Optional[dict]:
    g = _get_graph()
    if g is None:
        return None
    try:
        results = g.query(sparql)
        bindings = []
        for row in results:
            binding = {}
            for var in results.vars:
                val = row[var]
                if val is not None:
                    binding[str(var)] = {"value": str(val), "type": "uri"}
            bindings.append(binding)
        return {
            "results": {"bindings": bindings},
            "vars":    [str(v) for v in results.vars],
        }
    except Exception as e:
        return {"error": {"message": str(e)}}


# ── NL → SPARQL (LLM only, no hardcoded fallback) ────────────────────────────

def nl_to_sparql(question: str) -> str:
    """
    Convertit une question NL en SPARQL via le LLM.
    Applique clean_sparql() pour corriger les erreurs mineures.
    La validation et le repair sont gérés dans self_repair.py.
    """
    if not _check_ollama():
        raise RuntimeError("Ollama indisponible. Lance: ollama serve")

    raw    = generate_sparql_ollama(question)
    sparql = clean_sparql(raw)
    return sparql


# ── Demo ──────────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Which football players were born before 1920?",
    "Which players have French nationality?",
    "Which players play for Manchester United?",
    "Which players were born before 1990 and have United Kingdom nationality?",
    "List all football players born after 2000.",
]


def run_demo():
    print("=" * 60)
    print("SPARQL Generator — NL → SPARQL via LLM (rdflib local)")
    print("=" * 60)
    print(f"[INFO] LLM : {LLM_MODEL} ({'✓' if _check_ollama() else '✗ ABSENT'})")
    print(f"[INFO] KB  : {KB_PATH} ({'✓' if os.path.exists(KB_PATH) else '✗ absente'})\n")

    for q in EXAMPLE_QUESTIONS:
        print(f"[Q] {q}")
        try:
            sparql  = nl_to_sparql(q)
            ok, msg = validate_sparql_syntax(sparql)
            print(f"[SPARQL] {'✓ valide' if ok else '✗ ' + msg}")
            print(sparql.strip())
            if ok:
                results = execute_sparql(sparql)
                if results and "results" in results:
                    n = len(results["results"]["bindings"])
                    print(f"[RESULTS] {n} résultat(s)")
                    for row in results["results"]["bindings"][:3]:
                        vals = [v["value"].split("/")[-1].replace("_", " ")
                                for v in row.values()]
                        print(f"  → {' | '.join(vals)}")
                elif results and "error" in results:
                    print(f"[ERROR] {results['error']['message'][:150]}")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
        print("─" * 50)


if __name__ == "__main__":
    run_demo()