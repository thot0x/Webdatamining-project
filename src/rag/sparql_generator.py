"""
sparql_generator.py — NL → SPARQL via Ollama (gemma:2b or qwen2:0.5b)
Part of the RAG pipeline for the football knowledge graph.
"""
 
import os
import re
import json
import subprocess
import sys
from typing import Optional, Tuple
 
_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
 
#Configuration
 
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.environ.get("LLM_MODEL", "gemma:2b")   # or qwen2:0.5b on weak hw
SPARQL_ENDPOINT = os.environ.get(
    "SPARQL_ENDPOINT",
    "http://localhost:3030/football/sparql"   # Apache Jena Fuseki default
)
 
FOOTBALL_SCHEMA_SUMMARY = """
Prefixes:
  fb:  <http://example.org/football#>
  wdt: <http://www.wikidata.org/prop/direct/>
  wd:  <http://www.wikidata.org/entity/>
 
Main classes:
  fb:FootballPlayer  — a player (entity)
  fb:Club            — a football club
  fb:League          — a competition league
  fb:Country         — nationality / country
 
Key properties:
  fb:playsFor        FootballPlayer → Club
  fb:hasNationality  FootballPlayer → Country
  fb:competesIn      Club → League
  fb:hasBirthYear    FootballPlayer → xsd:integer
  fb:hasName         any → xsd:string
  fb:hasPosition     FootballPlayer → xsd:string  (Forward / Midfielder / Defender / Goalkeeper)
  fb:hasMarketValue  FootballPlayer → xsd:float
  owl:sameAs         links to Wikidata URIs
 
Example entities:
  fb:Cristiano_Ronaldo, fb:Lionel_Messi, fb:Kylian_Mbappe
  fb:Real_Madrid, fb:Man_City, fb:PSG
  fb:LaLiga, fb:PremierLeague, fb:Bundesliga
"""
 
SYSTEM_PROMPT = f"""You are an expert SPARQL query generator for a football knowledge graph.
Given a natural-language question, generate a syntactically correct SPARQL SELECT query.
 
Rules:
- Return ONLY the SPARQL query, no commentary, no markdown fences.
- Use the schema summary below.
- Always include PREFIX declarations at the top.
- Use OPTIONAL and FILTER where appropriate.
- Limit results to 20 unless asked otherwise.
 
Schema:
{FOOTBALL_SCHEMA_SUMMARY}
"""
 
 
#Ollama client
 
def _check_ollama():
    """Return True if Ollama is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False
 
 
def generate_sparql_ollama(question: str, model: str = LLM_MODEL) -> str:
    """Call Ollama /api/generate and return the generated SPARQL string."""
    import urllib.request
 
    payload = json.dumps({
        "model":  model,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nSPARQL:",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }).encode()
 
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
 
    sparql = data.get("response", "").strip()
    return sparql
 
 
# ── SPARQL post-processing ────────────────────────────────────────────────────
 
def clean_sparql(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    # Remove ```sparql ... ``` or ```...```
    raw = re.sub(r"```(?:sparql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()
    return raw
 
 
def validate_sparql_syntax(sparql: str) -> Tuple[bool, str]:
    """Basic syntax validation (does not require a SPARQL endpoint)."""
    lower = sparql.lower().strip()
    if not lower.startswith(("select", "prefix", "ask", "construct", "describe")):
        return False, "Query does not start with a valid SPARQL keyword."
    if "select" not in lower and "ask" not in lower:
        return False, "No SELECT or ASK clause found."
    if "where" not in lower:
        return False, "No WHERE clause found."
    return True, "OK"
 
 
# ── SPARQL execution ──────────────────────────────────────────────────────────
 
def execute_sparql(sparql: str, endpoint: str = SPARQL_ENDPOINT) -> Optional[dict]:
    """
    Execute SPARQL against a live SPARQL endpoint.
    Returns the JSON results dict or None on failure.
    """
    import urllib.request
    import urllib.parse
 
    params = urllib.parse.urlencode({"query": sparql, "format": "json"})
    url    = f"{endpoint}?{params}"
 
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None
 
 
# ── Prompt templates ──────────────────────────────────────────────────────────
 
EXAMPLE_QUESTIONS = [
    "Which players play for Real Madrid?",
    "Who are the French players in the knowledge graph?",
    "List all clubs that compete in the Premier League.",
    "Which players were born before 1990?",
    "What position does Kylian Mbappe play?",
]
 
FALLBACK_TEMPLATES = {
    "players for club": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {{
  ?player a fb:FootballPlayer ;
          fb:playsFor fb:{club} ;
          fb:hasName ?name .
}} LIMIT 20
""",
    "nationality": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {{
  ?player a fb:FootballPlayer ;
          fb:hasNationality "{nationality}" ;
          fb:hasName ?name .
}} LIMIT 20
""",
    "born before year": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name ?year WHERE {{
  ?player a fb:FootballPlayer ;
          fb:hasName ?name ;
          fb:hasBirthYear ?year .
  FILTER(?year < {year})
}} ORDER BY ?year LIMIT 20
""",
}
 
 
# ── Main interface ────────────────────────────────────────────────────────────
 
def nl_to_sparql(question: str, use_fallback: bool = False) -> str:
    """
    Convert a natural-language question to SPARQL.
    Falls back to template matching if Ollama is unavailable.
    """
    if not use_fallback and _check_ollama():
        try:
            raw    = generate_sparql_ollama(question)
            sparql = clean_sparql(raw)
            ok, msg = validate_sparql_syntax(sparql)
            if ok:
                return sparql
            print(f"[WARN] Generated query failed syntax check: {msg}")
        except Exception as e:
            print(f"[WARN] Ollama call failed: {e}")
 
    # Template fallback
    print("[INFO] Using template-based SPARQL generation.")
    q = question.lower()
    if "born before" in q:
        year = re.search(r"\d{4}", q)
        y    = year.group() if year else "1990"
        return FALLBACK_TEMPLATES["born before year"].format(year=y)
    if any(nat in q for nat in ["french", "spanish", "german", "english", "brazilian"]):
        mapping = {"french":"France","spanish":"Spain","german":"Germany",
                   "english":"England","brazilian":"Brazil"}
        nation  = next(v for k, v in mapping.items() if k in q)
        return FALLBACK_TEMPLATES["nationality"].format(nationality=nation)
    # Default: list all players
    return """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {
  ?player a fb:FootballPlayer ;
          fb:hasName ?name .
} LIMIT 20
"""
 
 
def run_demo():
    print("=" * 60)
    print("SPARQL Generator — NL → SPARQL via Ollama")
    print("=" * 60)
    print(f"[INFO] LLM model : {LLM_MODEL}")
    print(f"[INFO] Ollama URL: {OLLAMA_BASE_URL}")
    print(f"[INFO] Ollama available: {_check_ollama()}\n")
 
    for q in EXAMPLE_QUESTIONS:
        print(f"[Q] {q}")
        sparql = nl_to_sparql(q)
        print(f"[SPARQL]\n{sparql.strip()}\n{'─'*50}")
 
 
if __name__ == "__main__":
    run_demo()"""
sparql_generator.py — NL → SPARQL via Ollama (gemma:2b or qwen2:0.5b)
Part of the RAG pipeline for the football knowledge graph.
"""
 
import os
import re
import json
import subprocess
import sys
from typing import Optional, Tuple
 
_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
 
# ── Configuration ─────────────────────────────────────────────────────────────
 
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.environ.get("LLM_MODEL", "gemma:2b")   # or qwen2:0.5b on weak hw
SPARQL_ENDPOINT = os.environ.get(
    "SPARQL_ENDPOINT",
    "http://localhost:3030/football/sparql"   # Apache Jena Fuseki default
)
 
FOOTBALL_SCHEMA_SUMMARY = """
Prefixes:
  fb:  <http://example.org/football#>
  wdt: <http://www.wikidata.org/prop/direct/>
  wd:  <http://www.wikidata.org/entity/>
 
Main classes:
  fb:FootballPlayer  — a player (entity)
  fb:Club            — a football club
  fb:League          — a competition league
  fb:Country         — nationality / country
 
Key properties:
  fb:playsFor        FootballPlayer → Club
  fb:hasNationality  FootballPlayer → Country
  fb:competesIn      Club → League
  fb:hasBirthYear    FootballPlayer → xsd:integer
  fb:hasName         any → xsd:string
  fb:hasPosition     FootballPlayer → xsd:string  (Forward / Midfielder / Defender / Goalkeeper)
  fb:hasMarketValue  FootballPlayer → xsd:float
  owl:sameAs         links to Wikidata URIs
 
Example entities:
  fb:Cristiano_Ronaldo, fb:Lionel_Messi, fb:Kylian_Mbappe
  fb:Real_Madrid, fb:Man_City, fb:PSG
  fb:LaLiga, fb:PremierLeague, fb:Bundesliga
"""
 
SYSTEM_PROMPT = f"""You are an expert SPARQL query generator for a football knowledge graph.
Given a natural-language question, generate a syntactically correct SPARQL SELECT query.
 
Rules:
- Return ONLY the SPARQL query, no commentary, no markdown fences.
- Use the schema summary below.
- Always include PREFIX declarations at the top.
- Use OPTIONAL and FILTER where appropriate.
- Limit results to 20 unless asked otherwise.
 
Schema:
{FOOTBALL_SCHEMA_SUMMARY}
"""
 
 
# ── Ollama client ─────────────────────────────────────────────────────────────
 
def _check_ollama():
    """Return True if Ollama is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False
 
 
def generate_sparql_ollama(question: str, model: str = LLM_MODEL) -> str:
    """Call Ollama /api/generate and return the generated SPARQL string."""
    import urllib.request
 
    payload = json.dumps({
        "model":  model,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nSPARQL:",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }).encode()
 
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
 
    sparql = data.get("response", "").strip()
    return sparql
 
 
# ── SPARQL post-processing ────────────────────────────────────────────────────
 
def clean_sparql(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    # Remove ```sparql ... ``` or ```...```
    raw = re.sub(r"```(?:sparql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()
    return raw
 
 
def validate_sparql_syntax(sparql: str) -> Tuple[bool, str]:
    """Basic syntax validation (does not require a SPARQL endpoint)."""
    lower = sparql.lower().strip()
    if not lower.startswith(("select", "prefix", "ask", "construct", "describe")):
        return False, "Query does not start with a valid SPARQL keyword."
    if "select" not in lower and "ask" not in lower:
        return False, "No SELECT or ASK clause found."
    if "where" not in lower:
        return False, "No WHERE clause found."
    return True, "OK"
 
 
# ── SPARQL execution ──────────────────────────────────────────────────────────
 
def execute_sparql(sparql: str, endpoint: str = SPARQL_ENDPOINT) -> Optional[dict]:
    """
    Execute SPARQL against a live SPARQL endpoint.
    Returns the JSON results dict or None on failure.
    """
    import urllib.request
    import urllib.parse
 
    params = urllib.parse.urlencode({"query": sparql, "format": "json"})
    url    = f"{endpoint}?{params}"
 
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None
 
 
# ── Prompt templates ──────────────────────────────────────────────────────────
 
EXAMPLE_QUESTIONS = [
    "Which players play for Real Madrid?",
    "Who are the French players in the knowledge graph?",
    "List all clubs that compete in the Premier League.",
    "Which players were born before 1990?",
    "What position does Kylian Mbappe play?",
]
 
FALLBACK_TEMPLATES = {
    "players for club": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {{
  ?player a fb:FootballPlayer ;
          fb:playsFor fb:{club} ;
          fb:hasName ?name .
}} LIMIT 20
""",
    "nationality": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {{
  ?player a fb:FootballPlayer ;
          fb:hasNationality "{nationality}" ;
          fb:hasName ?name .
}} LIMIT 20
""",
    "born before year": """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name ?year WHERE {{
  ?player a fb:FootballPlayer ;
          fb:hasName ?name ;
          fb:hasBirthYear ?year .
  FILTER(?year < {year})
}} ORDER BY ?year LIMIT 20
""",
}
 
 
# ── Main interface ────────────────────────────────────────────────────────────
 
def nl_to_sparql(question: str, use_fallback: bool = False) -> str:
    """
    Convert a natural-language question to SPARQL.
    Falls back to template matching if Ollama is unavailable.
    """
    if not use_fallback and _check_ollama():
        try:
            raw    = generate_sparql_ollama(question)
            sparql = clean_sparql(raw)
            ok, msg = validate_sparql_syntax(sparql)
            if ok:
                return sparql
            print(f"[WARN] Generated query failed syntax check: {msg}")
        except Exception as e:
            print(f"[WARN] Ollama call failed: {e}")
 
    # Template fallback
    print("[INFO] Using template-based SPARQL generation.")
    q = question.lower()
    if "born before" in q:
        year = re.search(r"\d{4}", q)
        y    = year.group() if year else "1990"
        return FALLBACK_TEMPLATES["born before year"].format(year=y)
    if any(nat in q for nat in ["french", "spanish", "german", "english", "brazilian"]):
        mapping = {"french":"France","spanish":"Spain","german":"Germany",
                   "english":"England","brazilian":"Brazil"}
        nation  = next(v for k, v in mapping.items() if k in q)
        return FALLBACK_TEMPLATES["nationality"].format(nationality=nation)
    # Default: list all players
    return """
PREFIX fb: <http://example.org/football#>
SELECT ?player ?name WHERE {
  ?player a fb:FootballPlayer ;
          fb:hasName ?name .
} LIMIT 20
"""
 
 
def run_demo():
    print("=" * 60)
    print("SPARQL Generator — NL → SPARQL via Ollama")
    print("=" * 60)
    print(f"[INFO] LLM model : {LLM_MODEL}")
    print(f"[INFO] Ollama URL: {OLLAMA_BASE_URL}")
    print(f"[INFO] Ollama available: {_check_ollama()}\n")
 
    for q in EXAMPLE_QUESTIONS:
        print(f"[Q] {q}")
        sparql = nl_to_sparql(q)
        print(f"[SPARQL]\n{sparql.strip()}\n{'─'*50}")
 
 
if __name__ == "__main__":
    run_demo()