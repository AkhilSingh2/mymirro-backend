# MyMirro recommendation backend

A REST API behind the MyMirro styling app. It generates outfits for a user, finds similar outfits and products with embedding search, and recommends colours from a selfie or a picked skin-tone hex.

## Endpoints (prefix `/api/v1`, Swagger UI at `/swagger/`)
| Area | Endpoint |
|---|---|
| Outfits | `POST /outfits/generate`, `GET /outfits/{user_id}`, `GET /outfits/{outfit_id}/similar` |
| Products | `GET /products/{product_id}/similar` |
| Colour analysis | `POST /color/analyze`, `/color/analyze-photo`, `/color/analyze-hex` |
| Ops | `GET /health`, `/utils/warmup` |

## How it works
1. **Outfit generation** (phase 1): pulls the catalogue and user preferences from Supabase and assembles head-to-toe outfits with compatibility rules.
2. **Similar outfits** (phase 2): sentence-transformer embeddings and a FAISS index score style, colour, formality and pattern compatibility.
3. **Similar products** (phase 3): suggestions in the same category that vary in colour and design, weighted by the user's style, colour and price preferences.
4. **Skin-tone analysis**: OpenCV skin-region detection, Fitzpatrick type and colour recommendations by category.
5. **"Why we picked this"** explanations for each recommendation.
- Embeddings are precomputed (`precompute_embeddings.py`) and the model is cached for a quick start on Railway.

## Tech stack
Python · Flask, Flask-RESTX and Gunicorn · FAISS (CPU) · sentence-transformers and PyTorch · scikit-learn · OpenCV · pandas · Supabase · Docker · Railway

## Run locally
```bash
cp env.example .env      # SUPABASE_URL, SUPABASE_ANON_KEY, ...
pip install -r requirements.txt
python app.py
```

## More docs
- [docs/API_GUIDE.md](docs/API_GUIDE.md): the full feature and API guide
- [DEPLOY.md](DEPLOY.md): Railway deployment
- [docs/sample_outfits_response.json](docs/sample_outfits_response.json): an example response

## Credits
Built with Mayank Lakhani for MyMirro ([mymirro.in](https://www.mymirro.in)).
