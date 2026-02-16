#!/usr/bin/env python3
"""
Pre-compute ESCO embeddings to include in Docker image.
Run this locally, commit the embeddings, then Docker will use them.
"""
import os
import sys
from esco_taxonomy import ESCOTaxonomy

def main():
    print("=" * 60)
    print("Pre-computing ESCO embeddings...")
    print("=" * 60)
    
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        esco = ESCOTaxonomy(
            csv_path="data/esco/skills_en.csv",
            cache_path=os.path.join(cache_dir, "embedding_cache.pkl"),
            esco_cache_path=os.path.join(cache_dir, "esco_embeddings.pkl")
        )
        print("\n✓ ESCO embeddings pre-computed successfully!")
        print(f"✓ Saved to cache/esco_embeddings.pkl")
        print(f"✓ Size: {os.path.getsize(os.path.join(cache_dir, 'esco_embeddings.pkl')) / 1024 / 1024:.1f}MB")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
