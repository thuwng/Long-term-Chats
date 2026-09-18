"""
End-to-end MemORAI reproduction pipeline.

Usage
-----
python run_pipeline.py --dataset locomo --data_path data/locomo10.json \
    --limit 2 --out outputs/locomo_memorai.json

Ablations (Section 4.3 / Tables 4-8), pass any subset of:
    --no_topic_seg          -> Table 4 "w/o Topic Seg"
    --no_selective_filter   -> Table 4 "w/o Selective"
    --uniform_weight        -> Table 5 "Uniform (w=1)"
    --full_graph            -> Table 6/7 "Full Graph"
    --no_triplets           -> Table 8 "Turn only"

Baseline comparison (simple dense-retrieval baseline, no graph):
    --run_baseline

GPT-4o judge (costs real API calls, off by default):
    --run_judge
"""
import argparse
import logging
import time, json, os
from typing import List, Dict, Any

import numpy as np

from tqdm import tqdm
from config import DEFAULT_CONFIG, AblationConfig
from data.loaders import load_dataset
from llm_client import LLMClient, JudgeClient
from embeddings import EmbeddingModel
from indexing.build_graph import build_conversation_graph, graph_complexity_stats
from retrieval.seeding import get_seed_subgraph, get_full_graph_as_subgraph
from retrieval.dwpr import dynamic_weighted_pagerank
from retrieval.retrieve import select_top_turns, enrich_with_triplets, format_context
from generation.generate import generate_answer
from eval.recall import evaluate_retrieval
from eval.lexical_metrics import evaluate_generation
from eval.gpt4o_judge import gpt4o_judge_score
from utils import save_json, cosine_sim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="MemORAI reproduction pipeline")
    p.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--limit", type=int, default=None, help="limit #conversations (debug)")
    p.add_argument("--qa_limit", type=int, default=None, help="limit #QA per conversation (debug)")
    p.add_argument("--out", default="outputs/results.json")

    # ablations
    p.add_argument("--no_topic_seg", action="store_true")
    p.add_argument("--no_selective_filter", action="store_true")
    p.add_argument("--uniform_weight", action="store_true")
    p.add_argument("--full_graph", action="store_true")
    p.add_argument("--no_triplets", action="store_true")

    p.add_argument("--run_baseline", action="store_true", help="also run a dense-retrieval baseline")
    p.add_argument("--run_judge", action="store_true", help="also run GPT-4o-as-judge (costs $$)")
    p.add_argument("--resume", action="store_true",
                    help="skip conv_ids already present in <out>_raw.jsonl from a previous "
                         "(possibly Kaggle-timed-out) run, and append to it instead of "
                         "overwriting. Recommended for LOCOMO-10's full 10-conversation run.")
    return p.parse_args()


def _already_processed_conv_ids(raw_path: str) -> set:
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["conv_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def ablation_from_args(args) -> AblationConfig:
    return AblationConfig(
        use_topic_segmentation=not args.no_topic_seg,
        use_selective_filtering=not args.no_selective_filter,
        use_dynamic_weighting=not args.uniform_weight,
        use_subgraph_retrieval=not args.full_graph,
        use_triplet_enrichment=not args.no_triplets,
    )


def map_turns_to_sessions(turns: List[Dict[str, Any]]) -> Dict[str, str]:
    return {t["turn_id"]: t.get("session_id", t["turn_id"]) for t in turns}


def run_memorai_on_conversation(llm, embedder, cfg, ablation, conv):
    """Builds the graph once, then answers every QA pair for this conversation."""
    logger.info("  [graph] building knowledge graph for %s (%d raw turns)...",
                conv["conv_id"], len(conv["turns"]))
    t0 = time.time()
    G = build_conversation_graph(llm, embedder, conv["conv_id"], conv["turns"], ablation)
    logger.info("  [graph] done in %.1fs -> %d nodes / %d edges (structured errors: %d/%d)",
                time.time() - t0, G.number_of_nodes(), G.number_of_edges(),
                G.graph["stats"]["structured_output_errors"],
                max(G.graph["stats"]["structured_output_total"], 1))
    turn_to_session = map_turns_to_sessions(conv["turns"])

    records = []
    qa_iter = tqdm(conv["qas"], desc=f"  [qa] {conv['conv_id']}", leave=False)
    for qa_idx, qa in enumerate(qa_iter):
        q_emb = embedder.encode(qa["question"])

        if ablation.use_subgraph_retrieval:
            Gq = get_seed_subgraph(G, q_emb, k=cfg.retrieval.top_k_seed)
        else:
            Gq = get_full_graph_as_subgraph(G)

        if Gq.number_of_nodes() == 0:
            pred, ranked_turn_ids, ranked_session_ids = "The information provided is not enough", [], []
        else:
            pr_scores = dynamic_weighted_pagerank(
                Gq, q_emb, d=cfg.retrieval.damping, iters=cfg.retrieval.pagerank_iters,
                uniform=not ablation.use_dynamic_weighting,
            )
            top_turn_nodes = select_top_turns(Gq, pr_scores, m=cfg.retrieval.top_m_turns)
            triplets = enrich_with_triplets(Gq, top_turn_nodes) if ablation.use_triplet_enrichment else []
            context = format_context(Gq, top_turn_nodes, triplets,
                                      use_triplets=ablation.use_triplet_enrichment)
            pred = generate_answer(llm, context, qa["question"])

            all_turn_nodes_ranked = select_top_turns(Gq, pr_scores, m=len(Gq.nodes))
            ranked_turn_ids = [Gq.nodes[n]["turn_id"] for n in all_turn_nodes_ranked]
            seen_sessions, ranked_session_ids = set(), []
            for tid in ranked_turn_ids:
                sid = turn_to_session.get(tid, tid)
                if sid not in seen_sessions:
                    seen_sessions.add(sid)
                    ranked_session_ids.append(sid)

        records.append({
            "question": qa["question"], "reference": qa["answer"], "prediction": pred,
            "ranked_turn_ids": ranked_turn_ids, "gold_turn_ids": qa.get("gold_turn_ids", []),
            "ranked_session_ids": ranked_session_ids, "gold_session_ids": qa.get("gold_session_ids", []),
        })

    return records, graph_complexity_stats(G)


def run_dense_baseline_on_conversation(embedder, llm, cfg, conv):
    """
    Simple embedding-only baseline (~ 'Contriever' row in Table 1-3):
    rank all turns by cosine sim to the query, feed top-m verbatim turns
    (no graph, no triplets) straight into the same answer-generation prompt.
    """
    turns = conv["turns"]
    if not turns:
        return []
    turn_embs = embedder.encode([t["text"] for t in turns])
    turn_to_session = map_turns_to_sessions(turns)

    records = []
    for qa in conv["qas"]:
        q_emb = embedder.encode(qa["question"])
        sims = [cosine_sim(q_emb, e) for e in turn_embs]
        ranked_idx = np.argsort(sims)[::-1]
        ranked_turn_ids = [turns[i]["turn_id"] for i in ranked_idx]

        top_turns = [turns[i] for i in ranked_idx[: cfg.retrieval.top_m_turns]]
        context = "Relevant conversation turns:\n" + "\n".join(
            f"- [{t['turn_id']}{' (' + t['date'] + ')' if t.get('date') else ''}] {t['speaker']}: {t['text']}" for t in top_turns
        )
        pred = generate_answer(llm, context, qa["question"])

        seen_sessions, ranked_session_ids = set(), []
        for tid in ranked_turn_ids:
            sid = turn_to_session.get(tid, tid)
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                ranked_session_ids.append(sid)

        records.append({
            "question": qa["question"], "reference": qa["answer"], "prediction": pred,
            "ranked_turn_ids": ranked_turn_ids, "gold_turn_ids": qa.get("gold_turn_ids", []),
            "ranked_session_ids": ranked_session_ids, "gold_session_ids": qa.get("gold_session_ids", []),
        })
    return records


def summarize(records: List[Dict[str, Any]], judge_client=None) -> Dict[str, Any]:
    preds = [r["prediction"] for r in records]
    refs = [r["reference"] for r in records]

    gen_metrics = evaluate_generation(preds, refs) if preds else {}
    turn_recall = evaluate_retrieval(
        [r["ranked_turn_ids"] for r in records], [r["gold_turn_ids"] for r in records]
    )
    session_recall = evaluate_retrieval(
        [r["ranked_session_ids"] for r in records], [r["gold_session_ids"] for r in records]
    )

    out = {"generation": gen_metrics, "turn_recall": turn_recall, "session_recall": session_recall}
    if judge_client is not None and records:
        qs = [r["question"] for r in records]
        out["gpt4o_judge"] = gpt4o_judge_score(judge_client, qs, refs, preds)
    return out


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG

    ablation = ablation_from_args(args)

    logger.info("Loading dataset '%s' from %s", args.dataset, args.data_path)
    conversations = load_dataset(args.dataset, args.data_path)
    if args.limit:
        conversations = conversations[: args.limit]
    if args.qa_limit:
        for c in conversations:
            c["qas"] = c["qas"][: args.qa_limit]

    llm = LLMClient(cfg.llm)
    embedder = EmbeddingModel(cfg.embedding)
    judge_client = JudgeClient(cfg.judge) if args.run_judge else None

    all_memorai_records, all_baseline_records, graph_stats_list = [], [], []
    t0 = time.time()

    # --- THÊM ĐOẠN MỞ FILE RAW NGAY KHI CHẠY ---
    raw_path = args.out.rsplit(".", 1)[0] + "_raw.jsonl"
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)

    already_done = set()
    if args.resume:
        already_done = _already_processed_conv_ids(raw_path)
        if already_done:
            logger.info("Resuming: %d conv_ids already in %s, skipping them: %s",
                        len(already_done), raw_path, sorted(already_done))
            # Reload previously-written records so final metrics cover the
            # WHOLE dataset, not just conversations processed in this run
            # (important on Kaggle where a session can time out mid-run).
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    mode = rec.pop("mode", "memorai")
                    rec.pop("conv_id", None)
                    (all_memorai_records if mode == "memorai" else all_baseline_records).append(rec)

    raw_f = open(raw_path, "a" if args.resume else "w", encoding="utf-8")

    conv_iter = tqdm(conversations, desc="[conversations]")
    for conv in conv_iter:
        if conv["conv_id"] in already_done:
            continue
        logger.info("=== Processing conversation %s (%d QAs) ===", conv["conv_id"], len(conv["qas"]))
        conv_t0 = time.time()
        records, gstats = run_memorai_on_conversation(llm, embedder, cfg, ablation, conv)
        logger.info("  -> memorai: %d QA answered in %.1fs", len(records), time.time() - conv_t0)
        all_memorai_records.extend(records)
        graph_stats_list.append(gstats)

        # Ghi raw record ngay lập tức
        for r in records:
            raw_f.write(json.dumps({"conv_id": conv["conv_id"], "mode": "memorai", **r},
                                    ensure_ascii=False) + "\n")
        raw_f.flush()

        if args.run_baseline:
            base_t0 = time.time()
            baseline_records = run_dense_baseline_on_conversation(embedder, llm, cfg, conv)
            logger.info("  -> baseline: %d QA answered in %.1fs", len(baseline_records), time.time() - base_t0)
            all_baseline_records.extend(baseline_records)
            for r in baseline_records:
                raw_f.write(json.dumps({"conv_id": conv["conv_id"], "mode": "baseline", **r},
                                        ensure_ascii=False) + "\n")
            raw_f.flush()

    raw_f.close()
    logger.info("Đã lưu %d raw records (memorai) + %d (baseline) vào %s",
                len(all_memorai_records), len(all_baseline_records), raw_path)
    logger.info("Sinh câu trả lời xong toàn bộ trong %.1fs. Đang tính metric tổng hợp...", time.time() - t0)

    # 1. GIẢI PHÓNG VRAM GPU Ở ĐÂY ĐỂ TRÁNH OOM KHI TÍNH BERTSCORE
    import torch
    del llm
    del embedder
    torch.cuda.empty_cache()

    # 2. THÊM ĐIỀU KIỆN IF ELSE ĐỂ TRÁNH LỖI NAN KHI LIST RỖNG
    results = {
        "config": {
            "dataset": args.dataset,
            "ablation": ablation.__dict__,
            "llm_model": cfg.llm.model_name,
            "embedding_model": cfg.embedding.model_name,
        },
        "graph_complexity": {
            "avg_nodes": float(np.mean([g["n_nodes"] for g in graph_stats_list])) if graph_stats_list else 0.0,
            "avg_edges": float(np.mean([g["n_edges"] for g in graph_stats_list])) if graph_stats_list else 0.0,
        },
    }

    # Bọc try/except để tránh crash mất kết quả cuối
    try:
        results["memorai"] = summarize(all_memorai_records, judge_client)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Lỗi khi tính metric cho memorai: {e}. Xem raw records tại {raw_path}")
        results["memorai"] = {"error": str(e)}

    if args.run_baseline:
        try:
            results["dense_baseline"] = summarize(all_baseline_records, judge_client)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Lỗi khi tính metric cho baseline: {e}. Xem raw records tại {raw_path}")
            results["dense_baseline"] = {"error": str(e)}

    save_json(results, args.out)
    logger.info("Saved results to %s (raw per-QA records: %s)", args.out, raw_path)
    print_summary(results)


def print_summary(results: Dict[str, Any]):
    print("\n=== MemORAI reproduction summary ===")
    print(f"Graph complexity: {results['graph_complexity']}")
    for section in ("memorai", "dense_baseline"):
        if section not in results:
            continue
        print(f"\n-- {section} --")
        for group, metrics in results[section].items():
            print(f"  {group}: {metrics}")


if __name__ == "__main__":
    main()
