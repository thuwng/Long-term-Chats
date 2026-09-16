# MemORAI (unofficial reproduction)

Unofficial, from-scratch reimplementation of:

> **MemORAI: Memory Organization and Retrieval via Adaptive Graph
> Intelligence for LLM Conversational Agents** (ACL 2026 Findings)

No official code repository was found for this paper at the time of
writing, so this project reconstructs the pipeline directly from the
paper's methodology (Section 3) and the exact prompt templates in
Appendix C. Expect final numbers to differ somewhat from Tables 1-8
since several implementation details (exact chunking of long prompts,
PageRank convergence tolerance, top-k defaults) are not fully specified
in the paper.

## Pipeline overview

```
Phase 1 (indexing/segment.py, filter.py, summarize.py)
    raw conversation -> topical segments -> selective filtering -> segment summaries

Phase 2 (indexing/extract_triplets.py, extract_entities.py, build_graph.py)
    filtered segments -> entity-relation triplets + entity descriptions
    -> heterogeneous provenance graph G = (entities, turns, segments)

Phase 3 (retrieval/seeding.py, dwpr.py, retrieve.py; generation/generate.py)
    query -> multi-aspect seed nodes -> one-hop subgraph -> Dynamic
    Weighted PageRank -> top-m turns + supporting triplets -> LLM answer

Evaluation (eval/recall.py, lexical_metrics.py, gpt4o_judge.py)
    Recall@{3,5,10} (turn & session level), F1/BLEU/ROUGE/BERTScore, GPT-4o-as-judge
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in endpoints / keys, then `source .env` or export manually
```

You need:
1. A local OpenAI-compatible server hosting `openai/gpt-oss-20b`
   (e.g. `vllm serve openai/gpt-oss-20b --port 8000`), used for
   segmentation, filtering, summarization, extraction, and generation.
2. (Optional) an `OPENAI_API_KEY` for the GPT-4o judge (`--run_judge`).
3. LOCOMO-10 / LongMemEval-s data files — see `data/README.md`.

## Run

```bash
# Full MemORAI pipeline on a couple of LOCOMO-10 conversations, with baseline comparison
python run_pipeline.py --dataset locomo --data_path data/locomo10.json \
    --limit 2 --run_baseline --out outputs/locomo_memorai.json

# LongMemEval-s
python run_pipeline.py --dataset longmemeval --data_path data/longmemeval_s.jsonl \
    --limit 20 --out outputs/longmemeval_memorai.json
```

## Ablations (Section 4.3 / Tables 4-8)

```bash
python run_pipeline.py --dataset locomo --data_path data/locomo10.json --no_topic_seg
python run_pipeline.py --dataset locomo --data_path data/locomo10.json --no_selective_filter
python run_pipeline.py --dataset locomo --data_path data/locomo10.json --uniform_weight
python run_pipeline.py --dataset locomo --data_path data/locomo10.json --full_graph
python run_pipeline.py --dataset locomo --data_path data/locomo10.json --no_triplets
```

Combine flags freely; each toggles exactly the component named in the
corresponding ablation table.

## Notes on fidelity to the paper

- Prompts in `prompts/*.txt` are copied verbatim from Appendix C.1-C.7.
- Edge weighting in `retrieval/dwpr.py` implements Eq. 1 exactly (3 cases:
  entity->turn, entity->entity via relation, turn->segment).
- `config.py` centralizes `d` (damping), `top_k` (seed size), `top_m`
  (turns retrieved) — the paper does not report exact values for these,
  so tune them against Recall@k on a held-out split if you need to match
  Table 2/3 more closely.
- `data/loaders.py` field names may need small adjustments depending on
  the exact release version of LOCOMO-10 / LongMemEval-s you download
  (see the "ADAPT ME" note in that file).
