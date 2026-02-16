import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch

class ESCOTaxonomy:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.skill_names = self.df["preferredLabel"].astype(str).tolist()

        # Load semantic model
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.skill_embeddings = self.model.encode(
            self.skill_names,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def normalize(self, skills: set) -> dict:
        normalized = {}

        for skill in skills:

            # ---- 1️⃣ Try fast string match ----
            match = process.extractOne(
                skill,
                self.skill_names,
                scorer=fuzz.token_sort_ratio
            )

            if match and match[1] > 90:
                normalized[skill] = match[0]
                continue

            # ---- 2️⃣ Fallback to semantic match ----
            emb = self.model.encode(skill, convert_to_tensor=True)
            sims = util.cos_sim(emb, self.skill_embeddings)[0]

            best_idx = torch.argmax(sims).item()
            best_score = sims[best_idx].item()

            if best_score >= 0.70:
                normalized[skill] = self.skill_names[best_idx]
            else:
                normalized[skill] = skill

        return normalized
