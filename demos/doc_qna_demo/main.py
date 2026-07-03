"""
Doc-QnA with PaddleOCR-VL + OpenVINO
飞桨黑客松第10期 · 进阶任务 #13

一键端到端：构建索引 → RAG 问答 → 输出报告
所有推理均通过 OpenVINO 完成（Embedding + LLM）。

用法：
    python main.py                                   # 默认 5 题 demo
    python main.py --question "A100 工作温度？"       # 单条提问
    python main.py --device GPU                       # 切换设备
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows cmd 默认 cp1252，强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Windows 环境变量
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _ensure_dependencies():
    """检查关键依赖，缺失则自动 pip install"""
    try:
        import openvino  # noqa: F401
        import chromadb  # noqa: F401
        import huggingface_hub  # noqa: F401
        import jinja2  # noqa: F401  # transformers 5.x 不再传递依赖，chat_template 必需
    except ImportError:
        print("[自动安装] 检测到缺失依赖，正在执行 pip install -r requirements.txt ...")
        req = Path(__file__).parent / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req), "-q"])
        print("[自动安装] 依赖安装完成\n")


_ensure_dependencies()

from src.embedding import OpenVINOEmbedder, EmbedTiming  # noqa: E402
from src.llm import QwenLLM  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402
from src.vector_store import ChromaStore, iter_chunks_jsonl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("doc_qna_demo")


def parse_args():
    p = argparse.ArgumentParser(
        description="Doc-QnA Demo: PaddleOCR-VL + OpenVINO RAG Pipeline"
    )
    # 问题输入
    p.add_argument("--question", type=str, default=None,
                   help="单条问题（与 --questions_file 二选一）")
    p.add_argument("--questions_file", type=str, default="data/demo_questions.txt",
                   help="问题文件，每行一题（默认 5 题 demo）")

    # 数据
    p.add_argument("--chunks_dir", type=str, default="results/phase2",
                   help="Phase 2 输出的 chunks.jsonl 目录")
    p.add_argument("--persist_dir", type=str, default="chroma_db",
                   help="ChromaDB 持久化目录")

    # 设备（CPU / GPU / AUTO）
    p.add_argument("--device", type=str, default="CPU",
                   help="OpenVINO 推理设备：CPU / GPU / AUTO")

    # 模型
    p.add_argument("--embed_model_id", type=str,
                   default="OpenVINO/Qwen3-Embedding-0.6B-int8-ov")
    p.add_argument("--llm_model_id", type=str,
                   default="OpenVINO/Qwen3-1.7B-int4-ov")

    # 检索
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--min_score", type=float, default=0.35,
                   help="检索相似度下限，Top-K 全部低于该值时直接拒答（0 关闭）")
    p.add_argument("--max_new_tokens", type=int, default=384)

    # 输出（缺省时按运行模式选择，见 resolve_output_paths：
    # 5 题 demo → results/demo_run.*；单题 --question → results/demo_run_single.*，
    # 避免单题运行覆盖 5 题结果文件）
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out_md", type=str, default=None)
    return p.parse_args()


def resolve_output_paths(args):
    """--question 单题模式与默认 5 题 demo 使用不同的缺省输出文件，互不覆盖"""
    stem = "demo_run_single" if args.question else "demo_run"
    if args.out is None:
        args.out = f"results/{stem}.json"
    if args.out_md is None:
        args.out_md = f"results/{stem}.md"


def collect_questions(args):
    """收集问题列表"""
    if args.question:
        return [{"id": "q_cli", "question": args.question}]

    path = Path(args.questions_file)
    if not path.exists():
        logger.error(f"问题文件不存在: {path}")
        sys.exit(1)

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append({
                "id": f"q{len(questions)+1:03d}",
                "question": line,
            })
    return questions


def build_index(args, embedder):
    """构建或复用 ChromaDB 索引"""
    store = ChromaStore(persist_dir=args.persist_dir)

    # 检查是否已有数据
    if store.count() > 0:
        logger.info(f"ChromaDB 已有 {store.count()} 条，复用现有索引")
        return store

    # 从 chunks.jsonl 构建
    chunks_dir = Path(args.chunks_dir)
    files = sorted(chunks_dir.glob("*.chunks.jsonl"))
    if not files:
        logger.error(f"未找到 chunks 文件: {chunks_dir}/*.chunks.jsonl")
        sys.exit(1)

    chunks = iter_chunks_jsonl(files)
    logger.info(f"构建索引: {len(files)} 份文件, {len(chunks)} 个 chunks")

    store.reset_collection()
    texts = [c["text"] for c in chunks]
    timing = EmbedTiming()
    t0 = time.perf_counter()
    embs = embedder.encode(texts, batch_size=8, timing=timing)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"索引构建完成: {len(chunks)} chunks, dim={embs.shape[1]}, "
        f"{elapsed:.0f}ms ({elapsed/len(chunks):.1f} ms/chunk)"
    )
    store.add_chunks(chunks, embs)
    return store


def fmt_ms(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def run_qa(rag, questions):
    """运行问答并返回结果列表"""
    results = []
    t_run = time.perf_counter()

    for i, q in enumerate(questions, start=1):
        logger.info(f"[{i}/{len(questions)}] {q['question']}")
        ans = rag.answer(q["question"])

        print(f"\n{'='*78}")
        print(f"Q{i} ({q['id']}): {q['question']}")
        print(f"A: {ans.answer}")
        print(
            f"⏱  embed={fmt_ms(ans.timing.embed_query_ms)}ms  "
            f"retrieve={fmt_ms(ans.timing.retrieve_ms)}ms  "
            f"llm={fmt_ms(ans.timing.llm_ms)}ms  "
            f"total={fmt_ms(ans.timing.total_ms)}ms  "
            f"new_tokens={ans.timing.new_tokens}  "
            f"tps={ans.timing.tokens_per_second:.1f}"
        )
        print("Top-K 命中:")
        for h in ans.hits:
            print(f"  - {h.cite()} score={h.score:.3f}  "
                  f"{h.text[:100].replace(chr(10), ' ')}")

        results.append({**q, **ans.to_dict()})

    total_run = (time.perf_counter() - t_run) * 1000
    logger.info(f"全部完成: {len(results)} 题, {total_run:.0f}ms")
    return results, total_run


def save_results(args, results, total_run, store_count):
    """保存 JSON + Markdown 报告"""
    # JSON
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(results))
    out_json.write_text(json.dumps({
        "config": {
            "embed_model_id": args.embed_model_id,
            "llm_model_id": args.llm_model_id,
            "device": args.device,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": {
            "n_questions": len(results),
            "total_run_ms": round(total_run, 1),
            "avg_total_ms": round(sum(r["timing"]["total_ms"] for r in results) / n, 1),
            "avg_embed_ms": round(sum(r["timing"]["embed_query_ms"] for r in results) / n, 1),
            "avg_retrieve_ms": round(sum(r["timing"]["retrieve_ms"] for r in results) / n, 1),
            "avg_llm_ms": round(sum(r["timing"]["llm_ms"] for r in results) / n, 1),
            "avg_tps": round(sum(r["timing"]["tokens_per_second"] for r in results) / n, 1),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON → {out_json}")

    # Markdown
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Doc-QnA Demo · RAG 端到端问答结果",
        "",
        f"- Embedding: `{args.embed_model_id}` on `{args.device}`",
        f"- LLM: `{args.llm_model_id}` on `{args.device}`",
        f"- Top-K: `{args.top_k}` | max_new_tokens: `{args.max_new_tokens}`",
        f"- 向量库 chunks: {store_count}",
        "",
        "## 性能汇总",
        "",
        "| 阶段 | 平均耗时 (ms) |",
        "|------|---------------|",
        f"| embed query | {sum(r['timing']['embed_query_ms'] for r in results)/n:.1f} |",
        f"| retrieve | {sum(r['timing']['retrieve_ms'] for r in results)/n:.1f} |",
        f"| LLM | {sum(r['timing']['llm_ms'] for r in results)/n:.1f} |",
        f"| **total** | **{sum(r['timing']['total_ms'] for r in results)/n:.1f}** |",
        f"| LLM throughput | {sum(r['timing']['tokens_per_second'] for r in results)/n:.1f} tok/s |",
        "",
        "## 问答详情",
        "",
    ]
    for i, r in enumerate(results, start=1):
        lines += [
            f"### Q{i} ({r.get('id')})",
            "",
            f"**问题**: {r['question']}",
            "",
            f"**回答**: {r['answer'].strip()}",
            "",
            "**检索命中**:",
            "",
        ]
        for c in r["citations"]:
            lines.append(
                f"- `[{c['doc_name']} p.{c['page']}]` score={c['score']:.3f} "
                f"— {c['preview']}"
            )
        t = r["timing"]
        lines += [
            "",
            f"⏱ embed={t['embed_query_ms']}ms  retrieve={t['retrieve_ms']}ms  "
            f"llm={t['llm_ms']}ms  total={t['total_ms']}ms  "
            f"new_tokens={t['new_tokens']}  tps={t['tokens_per_second']}",
            "",
            "---",
            "",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown → {out_md}")


def main():
    args = parse_args()
    resolve_output_paths(args)

    # 收集问题
    questions = collect_questions(args)
    if not questions:
        logger.error("没有可跑的问题")
        sys.exit(1)

    print(f"\n{'='*78}")
    print("Doc-QnA Demo: PaddleOCR-VL + OpenVINO RAG Pipeline")
    print(f"{'='*78}")
    print(f"  Embedding: {args.embed_model_id}")
    print(f"  LLM:       {args.llm_model_id}")
    print(f"  Device:    {args.device}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*78}\n")

    # 初始化模型
    logger.info("加载 Embedding 模型...")
    embedder = OpenVINOEmbedder(
        model_id=args.embed_model_id,
        device=args.device,
    )

    # 构建/复用索引
    logger.info("准备向量索引...")
    store = build_index(args, embedder)

    logger.info("加载 LLM 模型...")
    llm = QwenLLM(
        model_id=args.llm_model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=False,
    )

    # RAG pipeline
    rag = RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        min_score=args.min_score,
    )

    # 运行问答
    results, total_run = run_qa(rag, questions)

    # 保存结果
    save_results(args, results, total_run, store.count())

    print(f"\n{'='*78}")
    print("Demo 完成!")
    print(f"  结果: {args.out}")
    print(f"  报告: {args.out_md}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
