"""
OpenVINO GenAI LLM 封装（Qwen3-1.7B-int4-ov）

通过 openvino_genai.LLMPipeline 直接加载 OpenVINO/Qwen3-1.7B-int4-ov 的 IR。
Qwen3 默认带 thinking 模式，会输出 <think>...</think>；本项目走问答主线，
统一关闭（enable_thinking=False），输出更紧凑。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "OpenVINO/Qwen3-1.7B-int4-ov"

# Qwen3 在 enable_thinking=False 下偶尔仍会自发输出 <think>...</think>，
# 这里强制剥离，只保留外部可见的最终答案。
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """
    剥离 Qwen3 的 <think>...</think> 块，处理三种典型形态：

    1. 完整 <think>真实思考</think>\\n\\n答案   → "答案"
    2. <think>\\n\\n答案   (/no_think 下模板预填 <think> 但模型直接给答案)
       → "答案"
    3. <think>未结束的思考(被 max_new_tokens 截断)  → 截断提示

    形态 2 在带 /no_think 的提示下很常见，不能粗暴扔掉 <think> 后面的内容。
    区分 2 和 3 用简易启发：开头若是 `\\n*` 或极短空白（≤ 8 char）则视为模板预填，
    其余视为真实思考块被截断。
    """
    # 形态 1：完整闭合 → 直接抽掉
    cleaned = _THINK_RE.sub("", text)

    if "<think>" in cleaned and "</think>" not in cleaned:
        before, after = cleaned.split("<think>", 1)
        # 模板预填型：<think> 后只有少量空白就接答案
        m_lead = re.match(r"\s{0,8}", after)
        lead_len = m_lead.end() if m_lead else 0
        body = after[lead_len:]
        if body.strip():
            # 形态 2：丢掉 <think> 标签，保留答案
            cleaned = (before + body).strip()
        else:
            # 形态 3：真的截断在 think 里
            return "[生成被截断在思考阶段，建议加大 max_new_tokens 或在 user 末尾加 /no_think]"

    return cleaned.strip()


@dataclass
class GenTiming:
    generate_ms: float = 0.0
    new_tokens: int = 0
    tokens_per_second: float = 0.0
    extras: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    timing: GenTiming = field(default_factory=GenTiming)


class QwenLLM:
    """
    用 openvino_genai.LLMPipeline 加载 Qwen3-1.7B-int4-ov。

    用法：
        llm = QwenLLM(device="CPU")
        resp = llm.chat([{"role":"user","content":"hi"}])
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "CPU",
        cache_dir: Optional[str | Path] = None,
        local_dir: Optional[str | Path] = None,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.temperature = temperature

        if local_dir is not None:
            ir_dir = Path(local_dir)
            if not (ir_dir / "openvino_model.xml").exists():
                raise FileNotFoundError(f"local_dir 不含 openvino_model.xml: {ir_dir}")
        else:
            from huggingface_hub import snapshot_download

            logger.info(f"下载 / 复用 LLM IR: {model_id}")
            ir_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
        self.ir_dir = ir_dir
        logger.info(f"LLM IR: {ir_dir} on {device}")

        import openvino_genai as ov_genai

        self.ov_genai = ov_genai
        # NPU 在 PromptLookup decoding 上不支持；CPU/GPU 走默认
        self.pipe = ov_genai.LLMPipeline(str(ir_dir), device=device)

        self.gen_config = ov_genai.GenerationConfig()
        self.gen_config.max_new_tokens = max_new_tokens
        if temperature == 0.0:
            self.gen_config.do_sample = False
        else:
            self.gen_config.do_sample = True
            self.gen_config.temperature = temperature
        # 关闭 EOS 提早截断仅当显式 think 模式才放开

    # ── 生成 ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> LLMResponse:
        """
        messages: [{"role":"system|user|assistant","content":str}, ...]
        走 transformers chat_template（IR 里带 tokenizer），保证 Qwen3 prompt 格式正确。

        Qwen3 关闭思考的官方做法：chat_template enable_thinking=False *并且* 在
        最后一条 user 消息追加 "/no_think"。仅靠 template 在某些指令复杂的场景
        模型仍会自发输出 <think>，双保险。
        """
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking

        msgs = [dict(m) for m in messages]  # 浅拷贝避免改原始
        if not thinking and msgs and msgs[-1].get("role") == "user":
            content = msgs[-1].get("content", "")
            if "/no_think" not in content:
                msgs[-1]["content"] = content.rstrip() + " /no_think"

        prompt = self._render_chat_template(msgs, enable_thinking=thinking)
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> LLMResponse:
        cfg = self.gen_config
        if max_new_tokens is not None:
            cfg = self.ov_genai.GenerationConfig()
            cfg.max_new_tokens = max_new_tokens
            cfg.do_sample = self.gen_config.do_sample
            if self.gen_config.do_sample:
                cfg.temperature = self.temperature

        t0 = time.perf_counter()
        out = self.pipe.generate(prompt, cfg)
        elapsed = (time.perf_counter() - t0) * 1000

        # openvino_genai 在不同版本里返回 str 或 DecodedResults
        text = out if isinstance(out, str) else getattr(out, "texts", [str(out)])[0]
        if hasattr(text, "__iter__") and not isinstance(text, str):
            text = list(text)[0]
        text = str(text)
        if not self.enable_thinking:
            text = strip_thinking(text)

        # 估算 token 数（用 tokenizer encode 末文本长度近似）
        new_tokens = 0
        try:
            tok = self._tokenizer()
            new_tokens = len(tok.encode(text))
        except Exception:
            new_tokens = max(1, len(text) // 2)

        tps = (new_tokens / (elapsed / 1000)) if elapsed > 0 else 0.0
        timing = GenTiming(
            generate_ms=elapsed,
            new_tokens=new_tokens,
            tokens_per_second=tps,
        )
        return LLMResponse(text=text, timing=timing)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _tokenizer(self):
        if not hasattr(self, "_tok_cached"):
            from transformers import AutoTokenizer

            self._tok_cached = AutoTokenizer.from_pretrained(str(self.ir_dir))
        return self._tok_cached

    def _render_chat_template(self, messages: List[dict], enable_thinking: bool) -> str:
        """
        Qwen3 chat_template 支持 enable_thinking=False 关闭 <think>。
        老版本 transformers 没有该开关时，回退到 add_generation_prompt=True。
        """
        tok = self._tokenizer()
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
