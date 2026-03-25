"""
swrl_family.py — SWRL rules on family.owl with OWLReady2
Rule: Person(?p) ∧ hasAge(?p, ?age) ∧ swrlb:greaterThan(?age, 60) → OldPerson(?p)
"""
 
import os
from datetime import date
from owlready2 import (
    get_ontology, Thing, DataProperty, ObjectProperty,
    AllDisjoint, FunctionalProperty, sync_reasoner_pellet, Imp
)
 
# Path relative to project root
_HERE = os.path.dirname(__file__)
FAMILY_OWL_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "..", "kg_artifacts", "family.owl")
)
 
 
# ── Ontology construction ─────────────────────────────────────────────────────
 
def build_family_ontology(onto):
    """Define classes, properties and sample individuals."""
    with onto:
        # Classes
        class Person(Thing):
            pass
 
        class OldPerson(Person):
            """Inferred class: person older than 60 years."""
            pass
 
        class Man(Person):
            pass
 
        class Woman(Person):
            pass
 
        AllDisjoint([Man, Woman])
 
        # Data properties
        class hasAge(DataProperty, FunctionalProperty):
            domain = [Person]
            range  = [int]
 
        class hasBirthYear(DataProperty, FunctionalProperty):
            domain = [Person]
            range  = [int]
 
        class hasName(DataProperty, FunctionalProperty):
            domain = [Person]
            range  = [str]
 
        # Object properties
        class hasParent(ObjectProperty):
            domain = [Person]
            range  = [Person]
 
        class hasChild(ObjectProperty):
            domain           = [Person]
            range            = [Person]
            inverse_property = hasParent
 
        # Sample individuals
        current_year = date.today().year
        rows = [
            ("Alice",   65, "w"),
            ("Bob",     72, "m"),
            ("Charlie", 45, "m"),
            ("Diana",   58, "w"),
            ("Eve",     80, "w"),
            ("Frank",   30, "m"),
        ]
        ind = {}
        for name, age, g in rows:
            cls     = Man if g == "m" else Woman
            p       = cls(name)
            p.hasAge       = age
            p.hasName      = name
            p.hasBirthYear = current_year - age
            ind[name] = p
 
        ind["Alice"].hasChild   = [ind["Charlie"]]
        ind["Bob"].hasChild     = [ind["Charlie"]]
        ind["Charlie"].hasChild = [ind["Frank"]]
 
    return onto
 
 
#SWRL rule
 
def add_swrl_rules(onto):
    """
    SWRL rule:
        Person(?p) ∧ hasAge(?p, ?age) ∧ swrlb:greaterThan(?age, 60)
        → OldPerson(?p)
    """
    with onto:
        rule = Imp()
        rule.set_as_rule(
            "Person(?p), hasAge(?p, ?age), "
            "swrlb:greaterThan(?age, 60) -> OldPerson(?p)"
        )
    return rule
 
 
#Manual fallback
 
def _apply_rule_manually(onto):
    """Apply the OldPerson rule without a DL reasoner (pure Python fallback)."""
    Person    = onto.search_one(iri="*Person")
    OldPerson = onto.search_one(iri="*OldPerson")
    if not (Person and OldPerson):
        print("[MANUAL] Classes Person/OldPerson not found.")
        return
    count = 0
    for p in list(Person.instances()):
        age = getattr(p, "hasAge", None)
        if age is not None and age > 60:
            if OldPerson not in p.is_a:
                p.is_a.append(OldPerson)
                count += 1
    print(f"[MANUAL] {count} person(s) classified as OldPerson.")
 
 
#Main
 
def run():
    print("=" * 60)
    print("SWRL Family Ontology — OWLReady2 Demo")
    print("=" * 60)
 
    onto_iri = "http://example.org/family.owl"
 
    if os.path.exists(FAMILY_OWL_PATH):
        print(f"[INFO] Loading existing ontology: {FAMILY_OWL_PATH}")
        onto = get_ontology(f"file://{FAMILY_OWL_PATH}").load()
    else:
        print("[INFO] Building family ontology from scratch...")
        onto = get_ontology(onto_iri)
        onto = build_family_ontology(onto)
        os.makedirs(os.path.dirname(FAMILY_OWL_PATH), exist_ok=True)
        onto.save(file=FAMILY_OWL_PATH, format="rdfxml")
        print(f"[SAVED] {FAMILY_OWL_PATH}")
 
    # Print individuals before reasoning
    Person = onto.search_one(iri="*Person")
    print("\n[BEFORE REASONING] Persons:")
    if Person:
        for p in Person.instances():
            print(
                f"  {p.name:12s}  "
                f"age={getattr(p, 'hasAge', '?'):>3}  "
                f"classes={[c.name for c in p.is_a]}"
            )
 
    # Add SWRL rule
    print(
        "\n[RULE] Person(?p) ∧ hasAge(?p,?age) ∧ "
        "swrlb:greaterThan(?age,60) → OldPerson(?p)"
    )
    add_swrl_rules(onto)
 
    # Reason
    print("\n[REASONING] Running Pellet reasoner...")
    try:
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True,
            )
        print("[REASONING] Done.")
    except Exception as exc:
        print(f"[WARNING] Pellet unavailable ({exc}). Applying rule manually...")
        _apply_rule_manually(onto)
 
    # Print results
    OldPerson = onto.search_one(iri="*OldPerson")
    print("\n[RESULT] OldPerson instances:")
    if OldPerson:
        old_list = list(OldPerson.instances())
        if old_list:
            for p in old_list:
                print(f"  ✓ {p.name}  (age={getattr(p, 'hasAge', '?')})")
        else:
            print("  (none — reasoner may not have fired)")
    else:
        print("  OldPerson class not found.")
 
    # Save inferred ontology
    inferred_path = FAMILY_OWL_PATH.replace(".owl", "_inferred.owl")
    onto.save(file=inferred_path, format="rdfxml")
    print(f"\n[SAVED] Inferred ontology → {inferred_path}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run()