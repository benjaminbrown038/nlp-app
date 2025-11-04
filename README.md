# nlp-app

# nlp-app


A clean, production-ready Python NLP service. By default it uses a tiny scikit-learn
baseline (fast, no heavy dependencies). Flip a switch to enable Hugging Face
Transformers for richer tasks (sentiment, NER, zero-shot).


## Quickstart
```bash
git clone <your-repo> nlp-app && cd nlp-app
make init # or: bash scripts/setup_env.sh 3.10
make run
```

Predict examples

curl -s -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"text":"I absolutely love this app!"}' | jq


# With Transformers (optional):
export ENABLE_TRANSFORMERS=1
pip install -r requirements-optional.txt
uvicorn app.main:app --reload


# Named Entity Recognition
curl -s -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"text":"Barack Obama was born in Hawaii.", "task":"ner"}' | jq


# Zero-shot classification
curl -s -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"text":"This movie was unexpectedly good.", "task":"zero-shot", "candidate_labels":["positive","negative","neutral"]}' | jq

