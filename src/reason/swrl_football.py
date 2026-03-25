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
 
 
#Build minimal football ontology (if ontology.ttl not available)
 
def build_football_ontology(onto):
    """Classes, properties and sample players."""
    with onto:
        # Classes
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
 
        #Data properties
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
 
        #Object properties
        class playsFor(ObjectProperty):
            domain = [FootballPlayer]; range = [Club]
 
        class competesin(ObjectProperty):
            domain = [Club]; range = [League]
 
        #Sample players (famous players across eras)
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
            p.playsFor = clubs_created[club_name]
 
    return onto
 
 
#SWRL rule
 
def add_swrl_rules(onto):
    """
    SWRL rule:
        FootballPlayer(?p) ∧ hasBirthYear(?p, ?y) ∧ swrlb:lessThan(?y, 1990)
        → VeteranPlayer(?p)
    """
    with onto:
        rule = Imp()
        rule.set_as_rule(
            "FootballPlayer(?p), hasBirthYear(?p, ?y), "
            "swrlb:lessThan(?y, 1990) -> VeteranPlayer(?p)"
        )
    return rule
 
 
#Manual fallback
 
def _apply_rule_manually(onto):
    FP = onto.search_one(iri="*FootballPlayer")
    VP = onto.search_one(iri="*VeteranPlayer")
    if not (FP and VP):
        print("[MANUAL] Classes not found.")
        return
    count = 0
    for p in list(FP.instances()):
        year = getattr(p, "hasBirthYear", None)
        if year is not None and year < 1990:
            if VP not in p.is_a:
                p.is_a.append(VP)
                count += 1
    print(f"[MANUAL] {count} player(s) classified as VeteranPlayer.")
 
 
#Extended rules demo
 
def add_extended_rules(onto):
    """
    Bonus rules:
      R2: FootballPlayer(?p) ∧ hasBirthYear(?p,?y) ∧ swrlb:greaterThanOrEqual(?y,1990)
          → YoungPlayer(?p)          [illustrative — class must be declared]
    """
    with onto:
        # Declare YoungPlayer if not already there
        YoungPlayer = onto.search_one(iri="*YoungPlayer")
        if YoungPlayer is None:
            FP = onto.search_one(iri="*FootballPlayer")
            class YoungPlayer(FP if FP else Thing):  # type: ignore
                """Player born in 1990 or later."""
                pass
 
        r2 = Imp()
        r2.set_as_rule(
            "FootballPlayer(?p), hasBirthYear(?p, ?y), "
            "swrlb:greaterThanOrEqual(?y, 1990) -> YoungPlayer(?p)"
        )
    return onto
 
 
#Main
 
def run():
    print("=" * 60)
    print("SWRL Football KB — OWLReady2 Demo")
    print("=" * 60)
 
    onto_iri = FOOTBALL_NS
 
    if os.path.exists(ONTOLOGY_PATH):
        print(f"[INFO] Loading: {ONTOLOGY_PATH}")
        onto = get_ontology(f"file://{ONTOLOGY_PATH}").load()
    else:
        print("[INFO] Building football ontology from scratch...")
        onto = get_ontology(onto_iri)
        onto = build_football_ontology(onto)
        os.makedirs(os.path.dirname(FOOTBALL_OWL_OUT), exist_ok=True)
        onto.save(file=FOOTBALL_OWL_OUT, format="rdfxml")
        print(f"[SAVED] {FOOTBALL_OWL_OUT}")
 
    FP = onto.search_one(iri="*FootballPlayer")
    print("\n[BEFORE REASONING] Football players:")
    if FP:
        for p in FP.instances():
            print(
                f"  {getattr(p,'hasName',p.name):25s}  "
                f"born={getattr(p,'hasBirthYear','?')}  "
                f"classes={[c.name for c in p.is_a]}"
            )
 
    print(
        "\n[RULE] FootballPlayer(?p) ∧ hasBirthYear(?p,?y) ∧ "
        "swrlb:lessThan(?y,1990) → VeteranPlayer(?p)"
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
        print(f"[WARNING] Pellet unavailable ({exc}). Applying rules manually...")
        _apply_rule_manually(onto)
 
    # Results
    VP = onto.search_one(iri="*VeteranPlayer")
    print("\n[RESULT] VeteranPlayer instances (born < 1990):")
    if VP:
        vet_list = list(VP.instances())
        if vet_list:
            for p in vet_list:
                print(
                    f"  ✓ {getattr(p,'hasName',p.name):25s}  "
                    f"born={getattr(p,'hasBirthYear','?')}"
                )
        else:
            print("  (none — check reasoner)")
 
    YP = onto.search_one(iri="*YoungPlayer")
    if YP:
        print("\n[RESULT] YoungPlayer instances (born >= 1990):")
        for p in YP.instances():
            print(
                f"  ✓ {getattr(p,'hasName',p.name):25s}  "
                f"born={getattr(p,'hasBirthYear','?')}"
            )
 
    inferred_path = FOOTBALL_OWL_OUT.replace(".owl", "_inferred.owl")
    onto.save(file=inferred_path, format="rdfxml")
    print(f"\n[SAVED] Inferred ontology → {inferred_path}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run()