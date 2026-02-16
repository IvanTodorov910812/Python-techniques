import json
from sentence_transformers import SentenceTransformer, util

class SkillTaxonomyNormalizer:
    def __init__(self, taxonomy_dict=None):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Basic starter taxonomy (replace later with ESCO JSON)
        self.taxonomy = taxonomy_dict or {
            "python": ["python", "python3", "py"],
            "machine learning": ["machine learning", "ml", "deep learning"],
            "data analysis": ["data analysis", "data analytics", "analytics"],
            "sql": ["sql", "mysql", "postgresql", "sqlite"],
            "excel": ["excel", "ms excel", "microsoft excel"],
        }

        self.canonical_skills = list(self.taxonomy.keys())
        self.embeddings = self.model.encode(self.canonical_skills, convert_to_tensor=True)

    def normalize(self, skills: set):
        normalized = set()

        for skill in skills:
            emb = self.model.encode(skill, convert_to_tensor=True)
            sims = util.cos_sim(emb, self.embeddings)[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])

            if best_score > 0.60:
                normalized.add(self.canonical_skills[best_idx])
            else:
                normalized.add(skill.lower())

        return normalized
