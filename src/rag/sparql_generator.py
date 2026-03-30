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
  rdfs:label         → lang string with @en tag (e.g. "Erling Haaland"@en)
  ex:birthDate       → xsd:date (e.g. "1987-06-24"^^xsd:date)
  ex:birthPlace      → ex:Place URI (e.g. ex:Leeds, ex:Paris, ex:Rosario)
  ex:hasNationality  → ex:Country URI (e.g. ex:France, ex:Spain, ex:Germany,
                        ex:Norway, ex:Brazil, ex:Portugal, ex:Poland,
                        ex:UnitedKingdom, ex:UnitedKingdomOfGreatBritainAndIreland)
                        WARNING: UK players use EITHER ex:UnitedKingdom OR
                        ex:UnitedKingdomOfGreatBritainAndIreland — always use
                        VALUES to cover both when querying UK nationality.
  ex:playsFor        → ex:Club URI — EXACT values: ex:ManchesterUnited,
                        ex:ManchesterCity, ex:Liverpool, ex:FcBarcelona,
                        ex:RealMadrid, ex:BayernMunich
  ex:hasPosition     → ex:Position URI (e.g. ex:Forward, ex:Midfielder,
                        ex:Defender, ex:Goalkeeper, ex:Winger, ex:WingHalf,
                        ex:Fullback, ex:Centreback, ex:AttackingMidfielder)
  ex:heightM         → decimal (e.g. 1.86)

IMPORTANT — exact club URIs (copy these exactly, case-sensitive):
  FC Barcelona  → ex:FcBarcelona   (NOT ex:FCBarcelona, NOT ex:Barcelona)
  Bayern Munich → ex:BayernMunich
  Real Madrid   → ex:RealMadrid
  Man United    → ex:ManchesterUnited
  Man City      → ex:ManchesterCity
  Liverpool     → ex:Liverpool

For exact label lookup use the @en tag: rdfs:label "Player Name"@en
For partial/case-insensitive match use: FILTER(CONTAINS(LCASE(STR(?label)), "name"))
"""

FEW_SHOT_EXAMPLES = """
Q: Which players play for Manchester United?
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?player ?label WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:playsFor ex:ManchesterUnited .
} LIMIT 20

Q: Which players have French nationality?
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?player ?label WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:hasNationality ex:France .
} LIMIT 20

Q: List all football players born after 2000.
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:birthDate ?date .
  FILTER(?date > "2000-01-01"^^xsd:date)
} ORDER BY DESC(?date) LIMIT 20

Q: Which football players were born before 1920?
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:birthDate ?date .
  FILTER(?date < "1920-01-01"^^xsd:date)
} ORDER BY ?date LIMIT 20

Q: Which players were born before 1990 and have United Kingdom nationality?
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?player ?label ?date WHERE {
  VALUES ?nat { ex:UnitedKingdom ex:UnitedKingdomOfGreatBritainAndIreland }
  ?player a ex:FootballPlayer ;
          rdfs:label ?label ;
          ex:hasNationality ?nat ;
          ex:birthDate ?date .
  FILTER(?date < "1990-01-01"^^xsd:date)
} ORDER BY ?date LIMIT 20
"""

SYSTEM_PROMPT = f"""You are a SPARQL generator. Output ONLY one SPARQL query, no explanation.
Use PREFIX declarations. FILTER goes inside WHERE. LIMIT goes after }}.
Club URIs: ex:ManchesterUnited ex:ManchesterCity ex:Liverpool ex:FcBarcelona ex:RealMadrid ex:BayernMunich
UK nationality: VALUES ?nat {{ ex:UnitedKingdom ex:UnitedKingdomOfGreatBritainAndIreland }}

{FEW_SHOT_EXAMPLES}"""


def _build_query_from_keywords(question: str) -> Optional[str]:
    """
    Build a SPARQL query by detecting key concepts in the question.
    Returns a valid query string, or None if no known patterns are detected
    (in which case the LLM fallback should be used).
    """
    q = question.lower()

    # Must be a player question
    if not any(w in q for w in ("player", "players", "footballer", "footballeur", "joueur")):
        return None

    predicates: list = []    # (property, value-or-var) pairs for the main triple
    extra_vars:  list = []   # extra SELECT variables
    filters:     list = []   # FILTER expressions (strings)
    values_block: str = ""   # optional VALUES block
    order_by:     str = ""
    need_xsd            = False

    # ── Club ─────────────────────────────────────────────────────────────────
    for kw, uri in [
        ("manchester united", "ex:ManchesterUnited"), ("man united",   "ex:ManchesterUnited"),
        ("manchester city",   "ex:ManchesterCity"),   ("man city",     "ex:ManchesterCity"),
        ("liverpool",         "ex:Liverpool"),
        ("fc barcelona",      "ex:FcBarcelona"),      ("barcelona",    "ex:FcBarcelona"),
        ("real madrid",       "ex:RealMadrid"),
        ("bayern munich",     "ex:BayernMunich"),     ("bayern",       "ex:BayernMunich"),
    ]:
        if kw in q:
            predicates.append(("ex:playsFor", uri))
            break

    # ── Nationality ───────────────────────────────────────────────────────────
    uk_question = "united kingdom" in q or "british" in q
    if uk_question:
        values_block = (
            "  VALUES ?nat { ex:UnitedKingdom ex:UnitedKingdomOfGreatBritainAndIreland }"
        )
        predicates.append(("ex:hasNationality", "?nat"))
    else:
        for kw, uri in [
            ("french",      "ex:France"),   ("spain",      "ex:Spain"),
            ("spanish",     "ex:Spain"),    ("german",     "ex:Germany"),
            ("portuguese",  "ex:Portugal"), ("brazilian",  "ex:Brazil"),
            ("norwegian",   "ex:Norway"),   ("polish",     "ex:Poland"),
            ("belgian",     "ex:Belgium"),  ("colombian",  "ex:Colombia"),
            ("argentinian", "ex:Argentina"),("italian",    "ex:Italy"),
        ]:
            if kw in q:
                predicates.append(("ex:hasNationality", uri))
                break

    # ── Date filters ──────────────────────────────────────────────────────────
    m_after  = re.search(r'born after[^0-9]*(\d{4})',  q)
    m_before = re.search(r'born before[^0-9]*(\d{4})', q)
    if m_after:
        year = m_after.group(1)
        predicates.append(("ex:birthDate", "?date"))
        filters.append(f'?date > "{year}-01-01"^^xsd:date')
        extra_vars.append("?date")
        order_by  = "ORDER BY DESC(?date)"
        need_xsd  = True
    elif m_before:
        year = m_before.group(1)
        predicates.append(("ex:birthDate", "?date"))
        filters.append(f'?date < "{year}-01-01"^^xsd:date')
        extra_vars.append("?date")
        order_by  = "ORDER BY ?date"
        need_xsd  = True

    # ── Position ──────────────────────────────────────────────────────────────
    for kw, uri in [
        ("goalkeeper", "ex:Goalkeeper"), ("forward",    "ex:Forward"),
        ("midfielder", "ex:Midfielder"), ("defender",   "ex:Defender"),
        ("winger",     "ex:Winger"),     ("striker",    "ex:Forward"),
    ]:
        if kw in q:
            predicates.append(("ex:hasPosition", uri))
            break

    # No structured concept found — fall back to LLM
    if not predicates and not values_block:
        return None

    # ── Assemble query ────────────────────────────────────────────────────────
    xsd_prefix = "\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>" if need_xsd else ""
    select_vars = "?player ?label" + ("".join(f" {v}" for v in extra_vars))

    # Build semicolon-chained triple
    triple_lines = ["  ?player a ex:FootballPlayer ;", "          rdfs:label ?label"]
    for prop, val in predicates:
        triple_lines.append(f"          {prop} {val}")
    # Last predicate ends with ' .'
    triple_lines[-1] += " ."
    # Others end with ' ;'
    for i in range(1, len(triple_lines) - 1):
        triple_lines[i] += " ;"

    where_parts = []
    if values_block:
        where_parts.append(values_block)
    where_parts.extend(triple_lines)
    for f in filters:
        where_parts.append(f"  FILTER({f})")

    body = "\n".join(where_parts)
    query = (
        f"PREFIX ex: <http://example.org/>\n"
        f"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>{xsd_prefix}\n"
        f"SELECT {select_vars} WHERE {{\n{body}\n}}"
    )
    if order_by:
        query += f" {order_by}"
    query += " LIMIT 20"
    return query


def _extract_hints(question: str) -> str:
    """Hint line injected into LLM prompt for questions the builder can't handle fully."""
    q = question.lower()
    parts = []
    for kw, uri in [
        ("manchester united", "ex:ManchesterUnited"), ("man united", "ex:ManchesterUnited"),
        ("manchester city",   "ex:ManchesterCity"),   ("man city",   "ex:ManchesterCity"),
        ("liverpool", "ex:Liverpool"), ("barcelona", "ex:FcBarcelona"),
        ("real madrid", "ex:RealMadrid"), ("bayern", "ex:BayernMunich"),
    ]:
        if kw in q:
            parts.append(f"ex:playsFor {uri}")
            break
    for kw, uri in [
        ("french","ex:France"),("spanish","ex:Spain"),("german","ex:Germany"),
        ("portuguese","ex:Portugal"),("brazilian","ex:Brazil"),("norwegian","ex:Norway"),
    ]:
        if kw in q:
            parts.append(f"ex:hasNationality {uri}")
            break
    if "united kingdom" in q or "british" in q:
        parts.append("VALUES ?nat { ex:UnitedKingdom ex:UnitedKingdomOfGreatBritainAndIreland }")
    m = re.search(r'born after[^0-9]*(\d{4})', q)
    if m:
        parts.append(f'FILTER(?date > "{m.group(1)}-01-01"^^xsd:date)')
    else:
        m = re.search(r'born before[^0-9]*(\d{4})', q)
        if m:
            parts.append(f'FILTER(?date < "{m.group(1)}-01-01"^^xsd:date)')
    return " ; ".join(parts)

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
    hints = _extract_hints(question)
    hint_line = f"# Use: {hints}\n" if hints else ""
    payload = json.dumps({
        "model":  model,
        "prompt": f"{SYSTEM_PROMPT}\nQ: {question}\n{hint_line}",
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 400},
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

def _extract_first_query(sparql: str) -> str:
    """
    Return only the first complete SPARQL query (stops at balanced closing brace
    + optional trailing modifiers: ORDER BY, GROUP BY, HAVING, LIMIT, OFFSET).
    Fixes "Expected end of text, found 'SELECT'" caused by the LLM appending a
    second query or example after the first one.
    """
    depth = 0
    found_open = False
    for i, ch in enumerate(sparql):
        if ch == '{':
            depth += 1
            found_open = True
        elif ch == '}' and found_open:
            depth -= 1
            if depth == 0:
                tail = sparql[i + 1:]
                m = re.match(
                    r'(\s*(ORDER\s+BY\b[^\n]*|GROUP\s+BY\b[^\n]*|'
                    r'HAVING\b[^\n]*|LIMIT\s+\d+|OFFSET\s+\d+))*',
                    tail, re.IGNORECASE,
                )
                return sparql[: i + 1 + (m.end() if m else 0)].strip()
    return sparql


def clean_sparql(raw: str) -> str:
    raw = re.sub(r"```(?:sparql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")
    raw = re.sub(r"PREFIXes?\s*:", "PREFIX", raw, flags=re.IGNORECASE)

    match = re.search(r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", raw, re.IGNORECASE)
    if match:
        raw = raw[match.start():]

    # Fix missing { after WHERE (causes "Expected SelectQuery, found '?'")
    raw = re.sub(r'\bWHERE\s*\n(\s*(?:\?|VALUES\b|FILTER\b|BIND\b|OPTIONAL\b))',
                 r'WHERE {\n\1', raw, flags=re.IGNORECASE)

    # Fix unbalanced braces — insert missing } before first modifier/LIMIT
    open_b  = raw.count('{')
    close_b = raw.count('}')
    if open_b > close_b:
        limit_m = re.search(r'\n((?:ORDER\s+BY\b|GROUP\s+BY\b|LIMIT\s+\d+).*)$',
                            raw, re.IGNORECASE | re.DOTALL)
        if limit_m:
            raw = raw[:limit_m.start()] + '\n}' + raw[limit_m.start():]
        else:
            raw = raw.rstrip() + '\n}'

    # Move FILTER placed after the closing brace back inside WHERE
    stray_filter = re.search(
        r'(\})\s*\n\s*(FILTER\s*\([^)]*\))\s*\n?\s*((?:ORDER\s+BY\b[^\n]*)?\s*(?:LIMIT\s+\d+)?)\s*$',
        raw, re.IGNORECASE,
    )
    if stray_filter:
        before   = raw[:stray_filter.start(1)]
        f_clause = stray_filter.group(2).strip()
        tail     = stray_filter.group(3).strip()
        raw = before + f"  {f_clause}\n}}"
        if tail:
            raw += f"\n{tail}"

    raw = re.sub(r"(LIMIT\s+\d+)\s*;", r"\1", raw)
    raw = re.sub(r"(LIMIT\s+\d+)\s*\.", r"\1", raw)

    inner_limit = re.search(r"(LIMIT\s+\d+)\s*\n(\s*}\s*)$", raw, re.MULTILINE)
    if inner_limit:
        limit_str   = inner_limit.group(1)
        close_brace = inner_limit.group(2)
        raw = raw[:inner_limit.start()] + close_brace.rstrip() + "\n" + limit_str

    # Truncate at the end of the first complete query (removes second appended query)
    raw = _extract_first_query(raw)

    return raw.strip()


def validate_sparql_semantics(question: str, sparql: str) -> Tuple[bool, str]:
    """Vérifie que la query correspond sémantiquement à la question."""
    q = question.lower()
    s = sparql.lower().replace(" ", "").replace("\n", "")
    if any(w in q for w in ("player", "players", "footballer", "footballeur", "joueur")):
        if "footballplayer" not in s:
            return False, "ex:FootballPlayer absent pour une question sur les joueurs."
    return True, "OK"


def validate_sparql_syntax(sparql: str) -> Tuple[bool, str]:
    lower = sparql.lower().strip()
    if not lower.startswith(("select", "prefix", "ask", "construct", "describe")):
        return False, "Ne commence pas par un mot-clé SPARQL valide."
    if "select" not in lower and "ask" not in lower:
        return False, "Pas de clause SELECT."
    if "where" not in lower:
        return False, "Pas de clause WHERE."
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


# ── NL → SPARQL ───────────────────────────────────────────────────────────────

def nl_to_sparql(question: str) -> str:
    if not _check_ollama():
        raise RuntimeError("Ollama indisponible. Lance: ollama serve")

    raw    = generate_sparql_ollama(question)
    sparql = clean_sparql(raw)
    return sparql