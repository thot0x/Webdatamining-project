# Football Knowledge Graph — Web Data Mining Project

A full end-to-end pipeline that builds a football Knowledge Graph from crawled web data and answers natural-language questions over it using a local RAG (Retrieval-Augmented Generation) system.

**Authors:** Thomas Wartelle, Marcel Yammine

---

## Overview

```
Web crawl → Entity extraction → Knowledge Graph → Entity linking → RAG (NL → SPARQL → KB)
```

Ask questions like:
- *"Which players have French nationality?"*
- *"Which goalkeepers play for Bayern Munich?"*
- *"List all football players born after 2000."*

and get answers directly from the KB — no hallucination, no internet lookup.

---

## Project Structure

```
.
├── data/
│   ├── crawler_output.jsonl          # Raw crawled pages
│   └── extracted_entities.csv        # Entities extracted by the IE module
│
├── kg_artifacts/
│   ├── private_kb.ttl                # Raw KB built from extracted entities
│   ├── private_kb_linked.ttl         # KB after entity linking (Wikidata URIs)
│   ├── private_kb_linked_aligned.ttl
│   ├── expanded_kb.ttl               # Final KB (~89k triples, enriched from Wikidata)
│   └── mapping_table.csv             # Private entity → Wikidata URI mappings
│
├── kge_data/                         # Formatted data for KGE model training
│
├── src/
│   ├── crawl/                        # Web crawler
│   ├── kg/                           # KB construction
│   ├── kge/                          # Knowledge Graph Embeddings
│   ├── reason/                       # OWL reasoning / inference
│   └── rag/
│       ├── sparql_generator.py       # NL → SPARQL (keyword builder + LLM fallback)
│       ├── self_repair.py            # SPARQL validation + LLM self-repair loop
│       └── demo_cli.py               # CLI demo (batch + interactive mode)
│
├── reports/
│   └── demo_results.json             # Saved results from the last demo run
│
└── notebooks/                        # Jupyter exploration notebooks
```

---

## Knowledge Graph

- **Format:** Turtle (`.ttl`)
- **Namespace:** `http://example.org/` (prefix `ex:`)
- **Size:** ~89,000 triples

### Classes

| Class | Description |
|---|---|
| `ex:FootballPlayer` | A football player |
| `ex:Club` | A football club |
| `ex:Country` | A country / nationality |
| `ex:Place` | A city or place of birth |

### Properties

| Property | Range | Example |
|---|---|---|
| `rdfs:label` | lang string `@en` | `"Erling Haaland"@en` |
| `ex:birthDate` | `xsd:date` | `"2000-07-21"^^xsd:date` |
| `ex:birthPlace` | `ex:Place` URI | `ex:Leeds` |
| `ex:hasNationality` | `ex:Country` URI | `ex:France` |
| `ex:playsFor` | `ex:Club` URI | `ex:ManchesterUnited` |
| `ex:hasPosition` | position URI | `ex:Goalkeeper` |
| `ex:heightM` | decimal | `1.86` |

### Exact Club URIs

| Club | URI |
|---|---|
| FC Barcelona | `ex:FcBarcelona` |
| Real Madrid | `ex:RealMadrid` |
| Bayern Munich | `ex:BayernMunich` |
| Manchester United | `ex:ManchesterUnited` |
| Manchester City | `ex:ManchesterCity` |
| Liverpool | `ex:Liverpool` |

> **UK nationality:** Players may have either `ex:UnitedKingdom` or `ex:UnitedKingdomOfGreatBritainAndIreland`. Always use a `VALUES` clause to cover both when querying.

---

## RAG Pipeline

### Architecture

```
Question (NL)
    │
    ▼
Keyword Builder  (sparql_generator.py)
    │  detects clubs, nationalities, dates, positions
    │  assembles a guaranteed-valid SPARQL query
    │
    ├── [recognized] ─────────────────────┐
    │                                     │
    └── [unrecognized] → LLM generation   │
              (Ollama / gemma:2b)          │
                   │                      │
                   ▼                      ▼
            SPARQL Validation ←───────────┘
            ├── Syntax  (rdflib parseQuery)
            └── Semantics (ex:FootballPlayer presence)
                   │
                   ▼
            Execute on KB  (rdflib, fully local)
                   │
             ┌─────┴──────┐
           Error         Success → Results
             │
             ▼
        LLM Self-Repair  (up to 5 attempts)
        error message fed back to LLM each round
```

### Generation Strategy

**Keyword Builder (primary) — deterministic, no LLM:**
- Club names → `ex:playsFor <URI>`
- Nationality adjectives (French, Spanish, Brazilian…) → `ex:hasNationality <URI>`
- "United Kingdom" / "British" → dual-URI `VALUES` block
- "born before/after [year]" → `FILTER` with correct direction and year extracted from the question
- Positions (goalkeeper, midfielder, defender…) → `ex:hasPosition <URI>`
- All combinations work together

**LLM Fallback — for unsupported patterns:**
- Model: `gemma:2b` via Ollama
- Few-shot prompted with 5 reference SPARQL queries
- Output cleaned by `clean_sparql()` — fixes missing braces, FILTER outside WHERE, double-SELECT
- Up to 5 self-repair attempts with the error message fed back to the LLM each time

---

## Setup

### Requirements

```bash
pip install rdflib
```

[Ollama](https://ollama.com) must be running locally with `gemma:2b` pulled:

```bash
ollama serve
ollama pull gemma:2b
```

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `gemma:2b` | Model to use |
| `SPARQL_ENDPOINT` | `local://rdflib` | Execution target (local only for now) |

---

## Running the Demo

### Batch mode — baseline vs RAG

```bash
python src/rag/demo_cli.py
```

Runs 5 benchmark questions with a side-by-side comparison of static baseline answers vs KB-backed RAG answers. Saves results to `reports/demo_results.json`.

### Interactive mode

```bash
python src/rag/demo_cli.py --interactive
```

Ask any football question in natural language. The pipeline answers from the KB.

### Self-repair pipeline only

```bash
python src/rag/self_repair.py
```

Runs the 5 demo questions with verbose repair logs (each attempt, status, error).

---

## Interactive Mode — Coverage

**Reliably handled by the keyword builder:**
- Club membership: *"Which players play for Liverpool?"*
- Nationality: *"Which French players…"*, *"Brazilian footballers…"*
- UK nationality: *"Players with United Kingdom nationality"*, *"British players"*
- Date filters: *"Players born before 1990"*, *"Born after 2000"*
- Position: *"Which goalkeepers…"*, *"All midfielders for Real Madrid"*
- Combinations: *"Spanish defenders born before 1995 playing for Barcelona"*

**Handled by LLM fallback (less reliable with gemma:2b):**
- Specific players: *"Where was Mbappé born?"*, *"How tall is Haaland?"*
- Country name instead of adjective: *"Players from Norway"* (vs "Norwegian players")
- Birthplace: *"Players born in Paris"*
- Less common nationalities: Ivorian, Uzbek, Ecuadorian…
- Club nicknames: "Barça", "Man U", "The Reds"

**Not in the KB — no answer possible:**
- Match statistics (goals, assists, minutes)
- Transfer history, contracts, salaries
- Match results or standings

---

## Entity Linking

Private entities were linked to Wikidata via URI alignment. Mappings are stored in `kg_artifacts/mapping_table.csv` (`Private Entity`, `External URI`, `Confidence`). The expanded KB was then enriched with Wikidata properties to fill attributes missing from the crawled data.

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| KB / SPARQL | rdflib |
| LLM inference | Ollama (gemma:2b) |
| Web scraping | requests, BeautifulSoup |
| Data | pandas, NumPy |
| KGE | PyKEEN |
| Notebooks | Jupyter |
