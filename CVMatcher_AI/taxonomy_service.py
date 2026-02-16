import json

with open("ESCO_taxonomy.json") as f:
    ESCO = json.load(f)

def normalize_title(raw_title):
    raw = raw_title.lower()
    best_match = max(ESCO, key=lambda e: similarity(raw, e["preferredLabel"]))
    return best_match

def similarity(a, b):
    """Synonym + n-gram overlap match for taxonomy normalization."""
    return sum(1 for x in a.split() if x in b)
