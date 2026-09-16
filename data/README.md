# Data

This folder is intentionally empty of raw data (license/redistribution).
Download the two benchmarks used in the paper and place them here:

- **LOCOMO-10** (Maharana et al., 2024): https://github.com/snap-stanford/locomo
  -> save as `data/locomo10.json`
- **LongMemEval-s** (Wu et al., 2025): https://github.com/xiaowu0162/LongMemEval
  -> save as `data/longmemeval_s.jsonl`

Then run:

```bash
python run_pipeline.py --dataset locomo --data_path data/locomo10.json
python run_pipeline.py --dataset longmemeval --data_path data/longmemeval_s.jsonl
```

If the raw file schema differs from what `data/loaders.py` expects
(dataset releases change field names over time), open one sample with
`python -c "import json; print(json.load(open('data/locomo10.json'))[0].keys())"`
and adjust the field-name constants marked `ADAPT ME` in `loaders.py`.
