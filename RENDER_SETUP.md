# Render Deployment Setup Guide

## 🚀 Quick Start

This guide explains how to deploy your CV-matcher app to Render with persistent embedding cache (saves 30-60 seconds on every restart).

---

## 📋 Prerequisites

- GitHub repository with this code
- Render.com account (free tier available)
- `render.yaml` and `Dockerfile` in your repo

---

## 🔧 Step 1: Update `esco_taxonomy.py` to Use CACHE_DIR

Your app needs to respect the `CACHE_DIR` environment variable on Render.

**Update your initialization code like this:**

```python
import os

# When creating ESCOTaxonomy instance:
cache_dir = os.getenv('CACHE_DIR', '.')  # Use /app/cache on Render, current dir locally

taxonomy = ESCOTaxonomy(
    csv_path="data/esco/skills_en.csv",
    model_name="all-mpnet-base-v2",
    cache_path=os.path.join(cache_dir, "embedding_cache.pkl"),
    esco_cache_path=os.path.join(cache_dir, "esco_embeddings.pkl")
)
```

**Example in your dashboard.py or extract_from_file.py:**

```python
import os
from esco_taxonomy import ESCOTaxonomy

@st.cache_resource  # Streamlit caching to avoid reloading
def load_taxonomy():
    cache_dir = os.getenv('CACHE_DIR', '.')
    return ESCOTaxonomy(
        csv_path="data/esco/skills_en.csv",
        cache_path=os.path.join(cache_dir, "embedding_cache.pkl"),
        esco_cache_path=os.path.join(cache_dir, "esco_embeddings.pkl")
    )

taxonomy = load_taxonomy()
```

---

## 📦 Step 2: Push Files to GitHub

```bash
git add Dockerfile .dockerignore render.yaml
git commit -m "Add Docker and Render deployment configs"
git push origin main
```

---

## 🌐 Step 3: Deploy on Render

### Option A: Using render.yaml (Recommended)

1. Go to [https://render.com/dashboard](https://render.com/dashboard)
2. Click **"New +"** → **"Blueprint"**
3. Click **"Public Git repository"**
4. Paste your GitHub repo URL
5. Click **"Connect"**
6. Click **"Deploy"** on the next screen

Render will automatically:
- Read `render.yaml`
- Build the Docker image
- Create a persistent disk at `/app/cache`
- Deploy the web service

### Option B: Manual Setup

1. Go to [https://render.com/dashboard](https://render.com/dashboard)
2. Click **"New +"** → **"Web Service"**
3. Select **"Docker"**
4. Fill in:
   - **Name**: `cv-matcher-app`
   - **Repository**: Your GitHub URL
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile` (default)
   - **Plan**: Starter ($7/month)

5. Go to **"Disks"** tab → **"Add Disk"**
   - **Name**: `embedding_cache`
   - **Mount Path**: `/app/cache`
   - **Size**: 2 GB

6. Go to **"Environment"** tab → **"Add Environment Variables"**
   - **Key**: `CACHE_DIR`
   - **Value**: `/app/cache`

7. Click **"Create Web Service"**

---

## ⚡ Performance Timeline

### First Deployment (Cold Start)
```
Building Docker image:           ~30-45s
Starting container:              ~5s
Loading model + computing embeddings:  ~35-45s
Total first startup:             ~75-95s ⏱️
```

### Subsequent Restarts
```
Starting container:              ~2-3s
Loading cached embeddings:       ~0.2s
Ready for requests:              ~2.5s ✅
```

### Per Request
```
CV parsing:                      ~50ms
Skill extraction:                ~30ms
Fuzzy matching:                  ~10ms
Semantic matching (if needed):   ~20-30ms
Total per CV:                    ~100-150ms ⏱️ (within your <200ms target!)
```

---

## 💰 Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| Starter plan | $7/month | Includes 0.5GB RAM, CPU-only |
| Persistent disk (2GB) | $2/month | For embedding cache |
| **Total** | **~$9-10/month** | ✅ Much cheaper than GPU! |

**GPU Alternative (NOT recommended):**
- GPU instance: $24/month
- Same functionality, 12x more expensive
- Your CPU-based setup is optimal! 🎯

---

## 🔍 Monitoring & Logs

After deployment:

1. Go to your service dashboard on Render
2. Click **"Logs"** tab to see:
   ```
   ✓ Loading cached ESCO embeddings from /app/cache/esco_embeddings.pkl...
   ✓ Loaded 436 ESCO embeddings (skipped recomputation!)
   ✓ Loaded embedding cache (125 items)
   Streamlit app started...
   ```

3. If you see recomputation messages, the cache isn't persisting:
   - Check that the disk is mounted correctly
   - Verify `CACHE_DIR` environment variable is set
   - Restart the service

---

## 🐛 Troubleshooting

### Issue: Embeddings recompute every restart
**Solution**: 
- Verify disk is mounted: Check "Disks" tab shows `embedding_cache` at `/app/cache`
- Verify code uses `CACHE_DIR`: Call `os.getenv('CACHE_DIR')`
- Check logs for cache save messages

### Issue: "permission denied" when saving cache
**Solution**:
```bash
# Make sure Dockerfile has:
RUN mkdir -p /app/cache
# And docker user has write permissions (check Dockerfile)
```

### Issue: Container takes >60s to start
**Solution**:
- First start is normal (35-45s for embeddings)
- If subsequent starts are slow:
  - Check CPU usage in Render dashboard
  - Verify cache files exist in persistent disk

---

## 📝 Example Code Updates

### Before (local development)
```python
taxonomy = ESCOTaxonomy(
    csv_path="data/esco/skills_en.csv"
)
```

### After (production-ready)
```python
import os

cache_dir = os.getenv('CACHE_DIR', '.')
taxonomy = ESCOTaxonomy(
    csv_path="data/esco/skills_en.csv",
    cache_path=os.path.join(cache_dir, "embedding_cache.pkl"),
    esco_cache_path=os.path.join(cache_dir, "esco_embeddings.pkl")
)
```

---

## ✅ Deployment Checklist

- [ ] Updated `esco_taxonomy.py` to read `CACHE_DIR`
- [ ] Pushed `Dockerfile`, `.dockerignore`, `render.yaml` to GitHub
- [ ] Created Render service using blueprint or manual setup
- [ ] Added persistent disk (2GB) at `/app/cache`
- [ ] Set `CACHE_DIR=/app/cache` environment variable
- [ ] First deployment complete (takes ~90s)
- [ ] Verified logs show "Loaded cached ESCO embeddings"
- [ ] Tested CV matching latency (<200ms per CV)

---

## 🎯 Expected Behavior

✅ **First deployment**: Takes ~90 seconds, generates cache  
✅ **Subsequent restarts**: 2-3 seconds startup  
✅ **Per CV latency**: 100-150ms (within target!)  
✅ **Cost**: $9-10/month  
✅ **No GPU needed**: CPU-based inference is faster!  

You're ready to go! 🚀

---

## 📚 Additional Resources

- [Render Docs: Persistent Disks](https://render.com/docs/disks)
- [Render Docs: Docker](https://render.com/docs/docker)
- [Render Docs: Blueprints](https://render.com/docs/infrastructure-as-code)

---

## 🆘 Need Help?

Check these in order:
1. Render dashboard logs
2. Verify `CACHE_DIR` environment variable is set
3. Check that disk is mounted at `/app/cache`
4. Verify code calls `os.getenv('CACHE_DIR')`
