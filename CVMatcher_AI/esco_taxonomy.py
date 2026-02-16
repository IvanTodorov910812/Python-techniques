import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

class ESCOEngine:
    def __init__(self, skills_path="data/esco/skills_en.csv"):
        self.model = SentenceTransformer("all-mpnet-base-v2")

        df = pd.read_csv(skills_path)
        df = df[df["preferredLabel"].notna()]

        self.skills = df["preferredLabel"].str.lower().unique().tolist()
        self.skill_embeddings = self.model.encode(
            self.skills,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def normalize_skills(self, extracted_skills: set, threshold=0.65):
        normalized = set()

        for skill in extracted_skills:
            emb = self.model.encode(skill, convert_to_tensor=True)
            sims = util.cos_sim(emb, self.skill_embeddings)[0]

            best_idx = torch.argmax(sims).item()
            best_score = sims[best_idx].item()

            if best_score >= threshold:
                normalized.add(self.skills[best_idx])

        return normalized
