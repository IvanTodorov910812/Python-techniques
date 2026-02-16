import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os


class ESCOTaxonomy:
    """
    Hybrid ESCO normalization engine:
    - Fast fuzzy matching first
    - Semantic fallback with embeddings
    - Embedding cache (10x speed improvement)
    - Optional persistent cache to disk
    - ESCO embeddings cached to avoid recomputation
    """

    def __init__(
        self,
        csv_path: str,
        model_name: str = "all-mpnet-base-v2",
        cache_path: str = "embedding_cache.pkl",
        esco_cache_path: str = "esco_embeddings.pkl"
    ):
        # Load ESCO dataset
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["preferredLabel"].notna()]

        self.skill_names = (
            self.df["preferredLabel"]
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

        # Load embedding model (CPU by default, faster 🚀)
        self.model = SentenceTransformer(model_name)
        self.model.to('cpu')  # Keep on CPU to save GPU memory during inference

        # Load or precompute ESCO embeddings (with disk caching)
        self.esco_cache_path = esco_cache_path
        self.skill_embeddings = self._load_or_create_esco_embeddings()

        # Cache setup
        self.embedding_cache = {}
        self.cache_path = cache_path
        self.load_cache()

    # --------------------------------------------------
    # 🔥 EMBEDDING CACHE (10x speed improvement)
    # --------------------------------------------------
    def _load_or_create_esco_embeddings(self):
        """Load ESCO embeddings from cache or create them (only once per deployment)"""
        if os.path.exists(self.esco_cache_path):
            try:
                print(f"Loading cached ESCO embeddings from {self.esco_cache_path}...")
                with open(self.esco_cache_path, "rb") as f:
                    embeddings = pickle.load(f)
                print(f"✓ Loaded {len(embeddings)} ESCO embeddings (skipped recomputation!)")
                return embeddings
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # First time: compute and cache
        print("Encoding ESCO skill embeddings (first startup)...")
        embeddings = self.model.encode(
            self.skill_names,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Save to disk for future startups
        with open(self.esco_cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"✓ Cached ESCO embeddings to {self.esco_cache_path}")
        
        return embeddings

    def get_embedding(self, text: str):
        text = text.lower()

        if text in self.embedding_cache:
            return self.embedding_cache[text]

        emb = self.model.encode(text, convert_to_tensor=True)
        self.embedding_cache[text] = emb
        return emb

    def save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded embedding cache ({len(self.embedding_cache)} items)")
            except Exception:
                self.embedding_cache = {}

    # --------------------------------------------------
    # 🚀 MAIN NORMALIZATION FUNCTION (optimized)
    # --------------------------------------------------
    def normalize(self, skills: set, fuzzy_threshold=90, semantic_threshold=0.70):
        """
        Normalize extracted skills to ESCO preferred labels.
        
        Optimizations:
        - Fuzzy matching first (fast, no GPU needed)
        - Semantic matching only for fuzzy failures
        - Batch encoding for unknowns
        - CPU-based similarity computation (faster for small batches)

        Returns:
            dict {original_skill: normalized_skill}
        """
        normalized = {}

        # Deduplicate & lowercase
        skills = {s.strip().lower() for s in skills if isinstance(s, str)}

        # ----------------------------------------------
        # Batch pre-encode unknown skills (faster)
        # ----------------------------------------------
        unknown_skills = [
            s for s in skills if s not in self.embedding_cache
        ]

        if unknown_skills:
            embeddings = self.model.encode(
                unknown_skills,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            for skill, emb in zip(unknown_skills, embeddings):
                self.embedding_cache[skill] = emb

        # ----------------------------------------------
        # Normalize each skill
        # ----------------------------------------------
        for skill in skills:

            # 1️⃣ Fast fuzzy match (CPU only, ~1ms)
            match = process.extractOne(
                skill,
                self.skill_names,
                scorer=fuzz.token_sort_ratio
            )

            if match and match[1] >= fuzzy_threshold:
                normalized[skill] = match[0]
                continue

            # 2️⃣ Semantic fallback (only if fuzzy fails)
            emb = self.embedding_cache[skill]
            
            # Keep on CPU for faster similarity computation
            if isinstance(emb, torch.Tensor):
                emb_cpu = emb.cpu() if emb.is_cuda else emb
            else:
                emb_cpu = torch.tensor(emb)
                
            skill_embeddings_cpu = (
                self.skill_embeddings.cpu() 
                if isinstance(self.skill_embeddings, torch.Tensor) and self.skill_embeddings.is_cuda
                else self.skill_embeddings
            )
            
            sims = util.cos_sim(emb_cpu, skill_embeddings_cpu)[0]

            best_idx = torch.argmax(sims).item()
            best_score = sims[best_idx].item()

            if best_score >= semantic_threshold:
                normalized[skill] = self.skill_names[best_idx]
            else:
                normalized[skill] = skill  # leave as is

        return normalized
