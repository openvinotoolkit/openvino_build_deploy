"""
答案数值接地校验（Answer-Value Grounding）—— 事后抗幻觉的最后一道闸

针对的失败：检索**召不回正确 chunk**、但召回了一个"主体对、内容不含答案"的 chunk
时，1.7B 小模型会硬答一个具体数值。典型是 Q5"GB/T 2423.1—2008 的实施日期"——
真答案 2009-10-01 所在段落被英文标题稀释，检索/重排都没命中，模型却拿封面页的
标准年份"2008"编出"2008 年 1 月 1 日"。前面的守卫（实体/重排/主体接地）都拦不住，
因为主体"GB/T 2423.1"确实在证据里，只是**答案里的日期**是凭空捏的。

做法（确定性、极保守，只在"答案里的硬事实证据里根本没有"时才改判拒答）：
  - 从**答案**里抽出硬事实：完整日期（年月日）+ 显著数值（带单位或多位数）；
  - 逐个在**检索证据**里核对（归一化后按数字串/日期元组匹配）；
  - 只要有一个硬事实在证据里找不到 → 判定为模型编造 → 把回答改成拒答。

只查"完整日期"和"显著数值"，不查裸年份（否则"GB/T 2423.1—2008"里的 2008 会误判），
也不查普通词句——把误伤域内正确答案的概率压到最低。
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

# 完整日期：2008年1月1日 / 2009-10-01 / 2009/10/01 / 2008.12.30
# 分隔符形式的年份限定为 19xx/20xx，否则标准号 "GB/T 2423.1-2008" 会被误当成日期
# (2423,1,20)，把本模块要保护的答案自己拒掉。年月日再做范围校验双保险。
_DATE_CN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DATE_SEP = re.compile(r"((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
# 显著数值：带单位（500W / -20℃ / 80GB / 3.3V ...）或 ≥2 位的独立数字
_NUM_UNIT = re.compile(
    r"-?\d+(?:\.\d+)?\s*(?:℃|°C|W|kW|GB|MB|TB|V|A|Hz|MHz|GHz|%|瓦|伏|安|度|米|mm|cm)",
    re.IGNORECASE,
)
_NUM_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_dates(text: str) -> Set[Tuple[int, int, int]]:
    out: Set[Tuple[int, int, int]] = set()
    for pat in (_DATE_CN, _DATE_SEP):
        for y, m, d in pat.findall(text):
            y, m, d = int(y), int(m), int(d)
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                out.add((y, m, d))
    return out


def _significant_numbers(text: str) -> Set[str]:
    """答案里需要接地的数字串：带单位的数，或 ≥2 位的独立数字（去掉正负号与前导零）。"""
    nums: Set[str] = set()
    for chunk in _NUM_UNIT.findall(text):
        for tok in _NUM_TOKEN.findall(chunk):
            nums.add(tok.lstrip("+-").lstrip("0") or "0")
    for tok in _NUM_TOKEN.findall(text):
        core = tok.lstrip("+-")
        if len(core.replace(".", "")) >= 2:  # 忽略孤立个位数，避免噪声误判
            nums.add(core.lstrip("0") or "0")
    return nums


_CITE_RE = re.compile(r"\[[^\]]*\]")
# 千分位分隔符：仅 ASCII 逗号 + 恰好 3 位数字组（1,200 / 1,200,000）。不能吃掉句读用的
# 中文逗号（"…30，2009…" 里那是两个日期的分隔，误删会把 30 和 2009 粘成 302009）。
_GROUP_SEP_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def _strip_grouping(s: str) -> str:
    return _GROUP_SEP_RE.sub("", s)


def check_answer_grounding(
    answer: str, evidence: str, question: str = ""
) -> Optional[str]:
    """
    返回 None 表示答案里的硬事实都能在证据里找到（放行）；
    否则返回那个"证据里没有"的硬事实（应改判拒答）。

    三条降噪规则，避免误伤域内正确答案：
      - 先剥掉答案里的 [文档名 p.页码] 引用标注，页码不是答案事实；
      - 去掉数字的千分位分隔符（证据里的 "1,200 W" 与答案 "1200W" 应视作同一数）；
      - 忽略"问题里本来就出现过"的日期/数字——那是模型复述题干的实体（如标准号
        GB/T 2423.1、型号 A300），不是它新造的事实，不该由证据来核实。
    """
    ans = _strip_grouping(_CITE_RE.sub(" ", answer))  # 去引用标注 + 千分位
    evidence = _strip_grouping(evidence)
    question = _strip_grouping(question)

    # 1) 完整日期：答案里每个"新"日期都必须作为日期出现在证据中
    ans_dates = extract_dates(ans) - extract_dates(question)
    if ans_dates:
        ev_dates = extract_dates(evidence)
        for y, m, d in ans_dates:
            if (y, m, d) not in ev_dates:
                return f"{y}年{m}月{d}日"

    # 2) 显著数值：答案里"新"数字串必须以数字形式出现在证据中
    ev_norm = {t.lstrip("+-").lstrip("0") or "0" for t in _NUM_TOKEN.findall(evidence)}
    q_nums = _significant_numbers(question)
    for n in _significant_numbers(ans):
        if n not in q_nums and n not in ev_norm:
            return n
    return None


def refusal_text(fact: str) -> str:
    return (
        f"文档中未提及。（生成答案中的关键事实 “{fact}” 无法在检索到的原文中核实，"
        f"为避免幻觉已改判拒答）"
    )
