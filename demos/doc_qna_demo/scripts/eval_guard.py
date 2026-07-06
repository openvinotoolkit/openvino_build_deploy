"""
抗幻觉守卫评测 · 重排 + 实体守卫的分离度验证

目的：用数据证明"域外实体 + 域内字段"的串台题（如"火星探测器的额定功率"）在
加了 cross-encoder 重排 + 实体一致性守卫后，能在进 LLM 前被正确拒答，且不误伤域内题。

本脚本**不加载 1.7B LLM**——它只跑到守卫决策为止（embed → 检索 → 重排 → 阈值/实体
判定），所以很快、确定性强，专门用来回归"分离度"这一个指标。

对每条问题，同时给出：
  - bi_top1     : bi-encoder 检索最高 cosine（旧方案唯一信号）
  - OLD 决策    : 仅用 --min_score 0.35 阈值时会怎么判（复现旧 Known Limitation）
  - rr_top1     : cross-encoder 重排最高 sigmoid（新方案主力信号）
  - entity      : 实体守卫是否命中伪实体
  - NEW 决策    : reranker + entity_gate 组合后的判定
  - 是否正确

用法：
  python scripts/eval_guard.py
  python scripts/eval_guard.py --out results/phase3/guard_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.embedding import OpenVINOEmbedder  # noqa: E402
from src.entity_gate import EntityGate, check_grounding  # noqa: E402
from src.reranker import OpenVINOReranker  # noqa: E402
from src.vector_store import ChromaStore, iter_chunks_jsonl  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def read_questions(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="抗幻觉守卫分离度评测（不含 LLM）")
    p.add_argument("--chunks_dir", default="results/phase2")
    p.add_argument("--persist_dir", default="chroma_db")
    p.add_argument("--in_domain", default="data/demo_questions.txt")
    p.add_argument("--out_domain", default="data/eval_out_of_domain.txt")
    p.add_argument("--device", default="CPU")
    p.add_argument("--embed_model_id", default="OpenVINO/Qwen3-Embedding-0.6B-int8-ov")
    p.add_argument("--reranker_model_id", default="OpenVINO/bge-reranker-base-int8-ov")
    p.add_argument("--reranker_local_dir", default=None,
                   help="直接指向本地 reranker IR 目录（跳过下载，加速本地验证）")
    p.add_argument("--retrieve_top_k", type=int, default=20)
    p.add_argument("--old_min_score", type=float, default=0.35,
                   help="旧方案 bi-encoder 阈值（复现旧 Known Limitation 的判定）")
    p.add_argument("--rerank_min_score", type=float, default=0.30)
    p.add_argument("--out", default="results/phase3/guard_eval.json")
    return p.parse_args()


def main():
    args = parse_args()

    embedder = OpenVINOEmbedder(model_id=args.embed_model_id, device=args.device)
    store = ChromaStore(persist_dir=args.persist_dir)
    chunks = iter_chunks_jsonl(sorted(Path(args.chunks_dir).glob("*.chunks.jsonl")))
    if store.count() == 0:
        if not chunks:
            print(f"[ERR] 向量库为空且找不到 {args.chunks_dir}/*.chunks.jsonl，先构建索引")
            sys.exit(1)
        store.reset_collection()
        embs = embedder.encode([c["text"] for c in chunks], batch_size=8)
        store.add_chunks(chunks, embs)
    gate = EntityGate.from_chunks(chunks)
    reranker = OpenVINOReranker(
        model_id=args.reranker_model_id,
        local_dir=args.reranker_local_dir,
        device=args.device,
    )

    cases = [(q, "in") for q in read_questions(Path(args.in_domain))] + \
            [(q, "ood") for q in read_questions(Path(args.out_domain))]

    rows = []
    for q, label in cases:
        ent = gate.check(q)
        qvec = embedder.encode_queries([q])[0]
        qr = store.query(q, qvec, top_k=args.retrieve_top_k)
        bi_top1 = max((h.score for h in qr.hits), default=0.0)
        rr_top1 = 0.0
        survivor_passages = []
        if qr.hits:
            order = reranker.rerank(q, [h.text for h in qr.hits])
            rr_top1 = float(order[0][1])
            # 与 RAGPipeline 完全一致：主体接地只跑在过阈值的 survivors[:5] 上，
            # 不是全体 reranked 的 top-5（否则评测会认证生产不复现的行为）
            survivor_passages = [
                qr.hits[i].text for i, s in order if s >= args.rerank_min_score
            ][:5]

        old_decision = "ANSWER" if bi_top1 >= args.old_min_score else "REFUSE"
        # 与 RAGPipeline 同序：实体码 → 重排阈值 → 主体接地
        ungrounded = check_grounding(q, survivor_passages) if survivor_passages else None
        if ent is not None:
            new_decision, guard = "REFUSE", f"entity:{ent[1]}"
        elif rr_top1 < args.rerank_min_score:
            new_decision, guard = "REFUSE", "reranker"
        elif ungrounded is not None:
            new_decision, guard = "REFUSE", f"subject:{ungrounded}"
        else:
            new_decision, guard = "ANSWER", "-"

        want = "ANSWER" if label == "in" else "REFUSE"
        rows.append({
            "q": q, "label": label, "bi_top1": round(bi_top1, 3),
            "old": old_decision, "rr_top1": round(rr_top1, 3),
            "guard": guard, "new": new_decision,
            "want": want, "ok": new_decision == want,
        })

    # ── 打印表格 ──
    print("\n" + "=" * 100)
    print(f"{'label':5} {'bi_top1':>7} {'OLD':>7} | {'rr_top1':>7} {'guard':>14} {'NEW':>7} {'ok':>3}  question")
    print("-" * 100)
    for r in rows:
        flag = "✓" if r["ok"] else "✗"
        print(f"{r['label']:5} {r['bi_top1']:7.3f} {r['old']:>7} | "
              f"{r['rr_top1']:7.3f} {r['guard']:>14} {r['new']:>7} {flag:>3}  {r['q']}")
    print("=" * 100)

    # ── 汇总 ──
    ind = [r for r in rows if r["label"] == "in"]
    ood = [r for r in rows if r["label"] == "ood"]
    in_ans = sum(r["new"] == "ANSWER" for r in ind)
    ood_ref = sum(r["new"] == "REFUSE" for r in ood)
    old_ood_ref = sum(r["old"] == "REFUSE" for r in ood)
    bi_in = [r["bi_top1"] for r in ind]
    bi_ood = [r["bi_top1"] for r in ood]
    rr_in = [r["rr_top1"] for r in ind]
    rr_ood = [r["rr_top1"] for r in ood]

    def rng(v):
        return f"[{min(v):.3f}, {max(v):.3f}]" if v else "[]"

    print(f"\n域内正确作答 : {in_ans}/{len(ind)}   (应为全部)")
    print(f"域外正确拒答 : {ood_ref}/{len(ood)}   (旧方案仅 {old_ood_ref}/{len(ood)})")
    print(f"分离度 bi-encoder : 域内 {rng(bi_in)}  vs 域外 {rng(bi_ood)}  "
          f"（重叠区间 → 单阈值拦不干净）")
    print(f"分离度 reranker   : 域内 {rng(rr_in)}  vs 域外 {rng(rr_ood)}  "
          f"（间隔越大越好拦）")
    all_ok = all(r["ok"] for r in rows)
    print(f"\n结论: {'✅ 全部判定正确，#2 串台问题已解决' if all_ok else '❌ 仍有误判，见上表 ✗ 行'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "embed_model_id": args.embed_model_id,
            "reranker_model_id": args.reranker_model_id,
            "retrieve_top_k": args.retrieve_top_k,
            "old_min_score": args.old_min_score,
            "rerank_min_score": args.rerank_min_score,
        },
        "summary": {
            "in_domain_answered": f"{in_ans}/{len(ind)}",
            "ood_refused_new": f"{ood_ref}/{len(ood)}",
            "ood_refused_old": f"{old_ood_ref}/{len(ood)}",
            "bi_in_range": rng(bi_in), "bi_ood_range": rng(bi_ood),
            "rr_in_range": rng(rr_in), "rr_ood_range": rng(rr_ood),
            "all_correct": all_ok,
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON → {out}")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
