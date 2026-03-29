"""
swrl_football.py — SWRL rules on the football KB with OWLReady2
Rule: FootballPlayer(?p) ∧ hasBirthYear(?p, ?y) ∧ swrlb:lessThan(?y, 1990)
      → VeteranPlayer(?p)
"""

import os
from owlready2 import (
    get_ontology, Thing, DataProperty, ObjectProperty,
    FunctionalProperty, sync_reasoner_pellet, Imp
)

_HERE = os.path.dirname(__file__)
ONTOLOGY_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "..", "kg_artifacts", "ontology.ttl")
)
FOOTBALL_OWL_OUT = os.path.abspath(
    os.path.join(_HERE, "..", "..", "kg_artifacts", "football_swrl.owl")
)

FOOTBALL_NS = "http://example.org/football#"


# ── Build minimal football ontology (if ontology.ttl not available) ───────────

def build_football_ontology(onto):
    """Classes, properties and sample players."""
    with onto:
        # ── Classes
        class Agent(Thing): pass

        class FootballPlayer(Agent):
            """A professional football (soccer) player."""
            pass

        class VeteranPlayer(FootballPlayer):
            """Player born before 1990 — inferred by SWRL rule."""
            pass

        class Club(Thing): pass
        class League(Thing): pass
        class Country(Thing): pass

        # ── Data properties
        class hasName(DataProperty, FunctionalProperty):
            domain = [Agent]; range = [str]

        class hasBirthYear(DataProperty, FunctionalProperty):
            domain = [FootballPlayer]; range = [int]

        class hasNationality(DataProperty, FunctionalProperty):
            domain = [FootballPlayer]; range = [str]

        class hasPosition(DataProperty, FunctionalProperty):
            domain = [FootballPlayer]; range = [str]

        class hasMarketValueEUR(DataProperty, FunctionalProperty):
            domain = [FootballPlayer]; range = [float]

        # ── Object properties
        class playsFor(ObjectProperty):
            domain = [FootballPlayer]; range = [Club]

        class competesin(ObjectProperty):
            domain = [Club]; range = [League]

        # ── Sample players (famous players across eras)
        players = [
            # (name, birth_year, nation, position, club_name)
            ("Cristiano_Ronaldo", 1985, "Portugal",    "Forward",    "Al_Nassr"),
            ("Lionel_Messi",      1987, "Argentina",   "Forward",    "Inter_Miami"),
            ("Luka_Modric",       1985, "Croatia",     "Midfielder", "Real_Madrid"),
            ("Kylian_Mbappe",     1998, "France",      "Forward",    "Real_Madrid"),
            ("Vinicius_Jr",       2000, "Brazil",      "Forward",    "Real_Madrid"),
            ("Erling_Haaland",    2000, "Norway",      "Forward",    "Man_City"),
            ("Xavi_Hernandez",    1980, "Spain",       "Midfielder", "Al_Sadd"),
            ("Iker_Casillas",     1981, "Spain",       "Goalkeeper", "Retired"),
            ("Thierry_Henry",     1977, "France",      "Forward",    "Retired"),
            ("Zinedine_Zidane",   1972, "France",      "Midfielder", "Retired"),
        ]

        clubs_created = {}
        for name, year, nation, pos, club_name in players:
            p              = FootballPlayer(name)
            p.hasName      = name.replace("_", " ")
            p.hasBirthYear = year
            p.hasNationality = nation
            p.hasPosition  = pos

            if club_name not in clubs_created:
                c = Club(club_name)
                c.hasName = club_name.replace("_", " ")
                clubs_created[club_name] = c
            p.playsFor.append(clubs_created[club_name])

    return onto


# ── SWRL rule ─────────────────────────────────────────────────────────────────

import owlready2.rule as _R

def _ensure_veteran_class(onto):
    """Crée VeteranPlayer dans l'ontologie si absent."""
    VP = onto.search_one(iri="*VeteranPlayer")
    if VP is None:
        FP = onto.search_one(iri="*FootballPlayer")
        with onto:
            class VeteranPlayer(FP if FP else Thing):
                """Joueur né avant 1990 — inféré par règle SWRL."""
                pass
        VP = onto.search_one(iri="*VeteranPlayer")
    return VP


def add_swrl_rules(onto):
    """
    Règle SWRL adaptée à ontology.ttl (ex:birthDate xsd:date) :

        FootballPlayer(?p) ∧ birthDate(?p, ?d)
        ∧ swrlb:lessThan(?d, "1990-01-01"^^xsd:date)
        → VeteranPlayer(?p)

    Note : swrlb:lessThan supporte xsd:date dans Pellet.
    En mode fallback Python, on parse la date manuellement.
    """
    FP         = onto.search_one(iri="*FootballPlayer")
    VP         = _ensure_veteran_class(onto)
    birthDate_p = onto.search_one(iri="*birthDate")

    # birthDate peut être absent si l'ontologie chargée est minimale
    if birthDate_p is None:
        # Déclarer la propriété manquante dans l'ontologie
        with onto:
            class birthDate(DataProperty, FunctionalProperty):
                namespace = onto
                domain    = [FP]
                range     = [str]   # str pour compatibilité sans xsd:date
        birthDate_p = onto.search_one(iri="*birthDate")

    with onto:
        imp   = Imp()
        var_p = _R.Variable("p")
        var_d = _R.Variable("d")

        # Corps : FootballPlayer(?p)
        ca1 = _R.ClassAtom()
        onto._set_obj_triple_spo(ca1.storid, _R.swrl_class_predicate, FP.storid)
        ca1.arguments.append(var_p)

        # Corps : birthDate(?p, ?d)
        dpa = _R.DatavaluedPropertyAtom()
        onto._set_obj_triple_spo(dpa.storid, _R.swrl_property_predicate, birthDate_p.storid)
        dpa.arguments.append(var_p)
        dpa.arguments.append(var_d)

        # Corps : swrlb:lessThan(?d, "1990-01-01")
        ba = _R.BuiltinAtom()
        ba.builtin = "lessThan"
        ba.arguments.append(var_d)
        ba.arguments.append("1990-01-01")

        # Tête : VeteranPlayer(?p)
        ca2 = _R.ClassAtom()
        onto._set_obj_triple_spo(ca2.storid, _R.swrl_class_predicate, VP.storid)
        ca2.arguments.append(var_p)

        imp.body = [ca1, dpa, ba]
        imp.head = [ca2]

    print(f"[SWRL] Règle : {repr(imp)}")
    return imp


# ── Manual fallback ───────────────────────────────────────────────────────────

def _parse_year(val) -> int | None:
    """Extrait l'année depuis un xsd:date ('YYYY-MM-DD'), int ou str."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val)
    try:
        return int(s[:4])
    except (ValueError, IndexError):
        return None


def _apply_rule_manually(onto):
    """
    Fallback Python : applique la règle VeteranPlayer (né avant 1990)
    en lisant ex:birthDate (xsd:date) ou ex:hasBirthYear (int).
    """
    FP = onto.search_one(iri="*FootballPlayer")
    VP = onto.search_one(iri="*VeteranPlayer") or _ensure_veteran_class(onto)
    if not FP:
        print("[MANUAL] Classe FootballPlayer introuvable.")
        return
    count = 0
    for p in list(FP.instances()):
        # Essayer birthDate d'abord, puis hasBirthYear
        bd  = getattr(p, "birthDate", None)
        bdy = getattr(p, "hasBirthYear", None)
        year = _parse_year(bd) or _parse_year(bdy)
        if year is not None and year < 1990:
            if VP not in p.is_a:
                p.is_a.append(VP)
                count += 1
    print(f"[MANUAL] {count} joueur(s) classifié(s) comme VeteranPlayer.")


# ── Extended rules demo ───────────────────────────────────────────────────────

def add_extended_rules(onto):
    """
    Règle bonus : FootballPlayer(?p) ∧ birthDate(?p,?d)
                  ∧ swrlb:greaterThanOrEqual(?d, "1990-01-01")
                  → YoungPlayer(?p)
    """
    FP          = onto.search_one(iri="*FootballPlayer")
    birthDate_p = onto.search_one(iri="*birthDate")
    if birthDate_p is None:
        print("[SWRL] birthDate introuvable, règle YoungPlayer ignorée.")
        return onto

    with onto:
        YoungPlayer = onto.search_one(iri="*YoungPlayer")
        if YoungPlayer is None:
            class YoungPlayer(FP if FP else Thing):  # type: ignore
                """Joueur né en 1990 ou après."""
                pass
            YoungPlayer = onto.search_one(iri="*YoungPlayer")

        imp2  = Imp()
        var_p = _R.Variable("p2")
        var_d = _R.Variable("d2")

        ca1 = _R.ClassAtom()
        onto._set_obj_triple_spo(ca1.storid, _R.swrl_class_predicate, FP.storid)
        ca1.arguments.append(var_p)

        dpa = _R.DatavaluedPropertyAtom()
        onto._set_obj_triple_spo(dpa.storid, _R.swrl_property_predicate, birthDate_p.storid)
        dpa.arguments.append(var_p)
        dpa.arguments.append(var_d)

        ba = _R.BuiltinAtom()
        ba.builtin = "greaterThanOrEqual"
        ba.arguments.append(var_d)
        ba.arguments.append("1990-01-01")

        ca2 = _R.ClassAtom()
        onto._set_obj_triple_spo(ca2.storid, _R.swrl_class_predicate, YoungPlayer.storid)
        ca2.arguments.append(var_p)

        imp2.body = [ca1, dpa, ba]
        imp2.head = [ca2]

    print(f"[SWRL] Règle bonus : {repr(imp2)}")
    return onto


# ── Main ──────────────────────────────────────────────────────────────────────

def _owlready_load(owl_path: str, onto_iri: str):
    """
    Charge un fichier .owl dans OWLReady2 via fileobj pour éviter
    le bug Windows avec les URI file:// (slash devant la lettre de lecteur).
    """
    onto = get_ontology(onto_iri)
    with open(owl_path, "rb") as f:
        onto.load(fileobj=f)
    return onto


def _load_or_build(onto_iri):
    """
    OWLReady2 ne lit pas le Turtle (.ttl) nativement.
    Si ontology.ttl existe → conversion TTL→RDF/XML via rdflib puis chargement.
    Si football_swrl.owl existe → chargement direct.
    Sinon → construction depuis zéro avec des joueurs de démo.
    """
    # 1. Conversion TTL → OWL via rdflib
    if os.path.exists(ONTOLOGY_PATH):
        converted = FOOTBALL_OWL_OUT
        try:
            from rdflib import Graph as RDFGraph
            print(f"[INFO] Conversion TTL → RDF/XML : {ONTOLOGY_PATH}")
            g = RDFGraph()
            g.parse(ONTOLOGY_PATH, format="turtle")
            os.makedirs(os.path.dirname(converted), exist_ok=True)
            g.serialize(destination=converted, format="xml")
            print(f"[INFO] Chargement converti : {converted}")
            return _owlready_load(converted, onto_iri)
        except Exception as e:
            print(f"[WARN] Conversion TTL échouée ({e}). Construction depuis zéro.")

    # 2. Charger football_swrl.owl s'il existe déjà
    if os.path.exists(FOOTBALL_OWL_OUT):
        print(f"[INFO] Chargement : {FOOTBALL_OWL_OUT}")
        return _owlready_load(FOOTBALL_OWL_OUT, onto_iri)

    # 3. Construire depuis zéro
    print("[INFO] Construction de l'ontologie football depuis zéro...")
    onto = get_ontology(onto_iri)
    onto = build_football_ontology(onto)
    os.makedirs(os.path.dirname(FOOTBALL_OWL_OUT), exist_ok=True)
    onto.save(file=FOOTBALL_OWL_OUT, format="rdfxml")
    print(f"[SAVED] {FOOTBALL_OWL_OUT}")
    return onto


def run():
    print("=" * 60)
    print("SWRL Football KB — OWLReady2 Demo")
    print("=" * 60)

    onto_iri = FOOTBALL_NS
    onto = _load_or_build(onto_iri)

    def _born(p):
        """Retourne la date/année de naissance depuis birthDate ou hasBirthYear."""
        bd = getattr(p, "birthDate", None)
        if bd:
            return str(bd)[:10]
        by = getattr(p, "hasBirthYear", None)
        return str(by) if by else "?"

    FP = onto.search_one(iri="*FootballPlayer")
    print("\n[BEFORE REASONING] Football players:")
    if FP:
        for p in FP.instances():
            print(
                f"  {getattr(p,'hasName',p.name):25s}  "
                f"born={_born(p)}  "
                f"classes={[c.name for c in p.is_a]}"
            )

    print(
        "\n[RULE] FootballPlayer(?p) ∧ birthDate(?p,?d) ∧ "
        "swrlb:lessThan(?d,'1990-01-01') → VeteranPlayer(?p)"
    )
    add_swrl_rules(onto)
    add_extended_rules(onto)

    print("\n[REASONING] Running Pellet reasoner...")
    try:
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True,
            )
        print("[REASONING] Done.")
    except Exception as exc:
        print(f"[WARNING] Pellet unavailable. Applying rules manually...")
        _apply_rule_manually(onto)

    # Results
    VP = onto.search_one(iri="*VeteranPlayer")
    print("\n[RESULT] VeteranPlayer instances (born < 1990):")
    if VP:
        vet_list = list(VP.instances())
        if vet_list:
            for p in vet_list:
                print(f"  ✓ {getattr(p,'hasName',p.name):25s}  born={_born(p)}")
        else:
            print("  (none — aucun individu avec birthDate dans l'ontologie chargée)")

    YP = onto.search_one(iri="*YoungPlayer")
    if YP:
        print("\n[RESULT] YoungPlayer instances (born >= 1990):")
        for p in YP.instances():
            print(f"  ✓ {getattr(p,'hasName',p.name):25s}  born={_born(p)}")

    inferred_path = FOOTBALL_OWL_OUT.replace(".owl", "_inferred.owl")
    onto.save(file=inferred_path, format="rdfxml")
    print(f"\n[SAVED] Inferred ontology → {inferred_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()