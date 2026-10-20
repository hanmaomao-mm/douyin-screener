# -*- coding: utf-8 -*-
"""从 OCR 结果里解析：达人昵称、话题标签、看后搜词条、互动数据。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ocr import TextLine

# ------------------------------------------------------------------ 正则
RE_HASHTAG = re.compile(r"[#＃]\s*([^\s#＃，,。.！!？?、；;：:|\-—～~【】\[\]（）()]+)")
RE_AT = re.compile(r"[@＠]\s*([^\s@＠]{1,24})")
RE_PLAIN_NUM = re.compile(r"^\s*([\d]+(?:\.\d+)?)\s*([万wW千kK亿]?)\s*$")
RE_LIKE_INLINE = re.compile(r"([\d]+(?:\.\d+)?)\s*([万wW千kK亿]?)\s*(?:赞|点赞)")

_CN_UNIT = {"万": 1e4, "w": 1e4, "W": 1e4, "千": 1e3, "k": 1e3, "K": 1e3, "亿": 1e8}

# 右侧互动栏里常见的噪声文本，避免被当成数据
_METRIC_NOISE = {"点赞", "评论", "收藏", "分享", "转发", "下载", "复制链接", "举报", "不感兴趣",
                 "关注", "已关注", "音乐", "听歌", "截图", "倍速", "全屏", "发送", "说点什么"}

_SPAM_CHARS = re.compile(r"[\s\U0001F300-\U0001FAFF\u2600-\u27BF~～!！?？。．.,，、;；:：'\"“”‘’()（）\[\]【】<>《》\-_=+*/\\|`^$#@&%]+")


def parse_count(text: str) -> Optional[int]:
    """把 '1.2万' / '3.4w' / '890' 这类文本转成整数。"""
    if not text:
        return None
    t = text.strip().replace(",", "").replace("，", "").replace(" ", "")
    m = RE_PLAIN_NUM.match(t)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2)
    if unit:
        val *= _CN_UNIT.get(unit, 1.0)
    return int(round(val))


# ------------------------------------------------------------------ 话题标签
def extract_hashtags(lines: List[TextLine]) -> List[str]:
    tags: List[str] = []
    seen = set()
    for ln in lines:
        for m in RE_HASHTAG.finditer(ln.text):
            tag = m.group(1).strip().strip("#＃")
            if not tag or len(tag) > 20:
                continue
            k = tag.casefold()
            if k not in seen:
                seen.add(k)
                tags.append(tag)
    return tags


# ------------------------------------------------------------------ 看后搜
def extract_search_terms(lines: List[TextLine], prefixes: List[str],
                         heuristic: bool = True) -> Tuple[List[str], List[str]]:
    """返回 (确定命中, 推测命中)。

    - 确定：OCR 文本里带「看后搜 / 大家都在搜 / 搜一搜」等前缀
    - 推测：无前缀，但位于 caption 区的短文本行，疑似搜索引导词
    """
    confirmed: List[str] = []
    guessed: List[str] = []
    seen_c, seen_g = set(), set()
    prefix_re = re.compile("|".join(re.escape(p) for p in prefixes), re.IGNORECASE) if prefixes else None

    for ln in lines:
        raw = ln.text.strip()
        if not raw:
            continue
        # 顶部搜索框 / 窗口按钮（y < 0.07）不算看后搜，避免把"搜索你感兴趣的内容"误收进来
        if ln.frame_center_y < 0.07:
            continue
        matched_prefix = False
        if prefix_re:
            for m in prefix_re.finditer(raw):
                tail = raw[m.end():]
                tail = re.sub(r"^[\s:：，,\.。\-—~～]+", "", tail).strip()
                for piece in re.split(r"[，,、;；|/]", tail):
                    piece = piece.strip().strip("#＃")
                    if 1 < len(piece) <= 20:
                        k = piece.casefold()
                        if k not in seen_c:
                            seen_c.add(k)
                            confirmed.append(piece)
                        matched_prefix = True
        if matched_prefix:
            continue
        if not heuristic:
            continue
        if ln.region not in ("caption", "full"):
            continue
        # 启发式：短、无标点、不像昵称/无 @、不含 #，才当作疑似搜索词
        cand = raw.strip()
        if not (2 <= len(cand) <= 14):
            continue
        if "#" in cand or "＃" in cand or "@" in cand or "＠" in cand:
            continue
        if cand in _METRIC_NOISE:
            continue
        if re.search(r"\d{2,}|昨天|前天|小时前|分钟前|刚刚|·|：|:", cand):
            continue
        if re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", cand):
            k = cand.casefold()
            if k not in seen_g and k not in seen_c:
                seen_g.add(k)
                guessed.append(cand)
    return confirmed, guessed


# ------------------------------------------------------------------ 昵称
def extract_nickname(lines: List[TextLine]) -> str:
    """定位达人昵称。

    优先级：底部信息区（y>0.6 且 x<0.68）的 @ 提及 > profile 区纯文本 > 任意 @。
    @ 提及是最强特征，且不受窗口大小 / 布局漂移影响，所以放最前面。
    """
    cands = [ln for ln in lines if ln.frame_box[1] > 0.60 and ln.frame_box[0] < 0.68]
    cands.sort(key=lambda l: (l.frame_box[1], l.frame_box[0]))
    for ln in cands:
        m = RE_AT.search(ln.text)
        if m:
            return m.group(1).strip()

    profile_lines = [ln for ln in lines if ln.region == "profile"]
    profile_lines.sort(key=lambda l: (l.box[1], l.box[0]))
    for ln in profile_lines:
        raw = ln.text.strip()
        if 2 <= len(raw) <= 24 and not raw.startswith("#") and not raw.startswith("＃"):
            if re.search(r"[\u4e00-\u9fa5A-Za-z]", raw) and not re.search(r"\d{1,2}:\d{2}", raw):
                return raw

    for ln in lines:
        m = RE_AT.search(ln.text)
        if m:
            return m.group(1).strip()
    return ""


# ------------------------------------------------------------------ 互动数据
@dataclass
class Metrics:
    like: Optional[int] = None
    comment: Optional[int] = None
    collect: Optional[int] = None
    share: Optional[int] = None
    play: Optional[int] = None
    like_rate: Optional[float] = None
    like_rate_note: str = ""
    raw_texts: List[str] = field(default_factory=list)


def extract_metrics(rail_lines: List[TextLine], all_lines: List[TextLine],
                    comments: List[TextLine]) -> Metrics:
    """提取互动数据。

    抖音 PC 客户端的互动栏在画面右侧，纵向排列：点赞、评论、收藏、分享。
    数字有可能被 rail 裁剪区或 full 整幅区识别到，这里把两边的结果合并去重，
    再按纵坐标排序后依次归类，并在文本里出现「赞/评论」字样时按语义纠正。
    """
    m = Metrics()

    # ---- 数据来源选择 ----
    # 优先用「整幅画面右侧的文字」：它不会因为裁剪框正好切在数字中间而读错。
    # 只有当整幅画面在右侧识别到的数字不足 2 个时，才把 rail 裁剪区的结果合并进来兜底。
    def _numeric(lines: List[TextLine]) -> List[Tuple[float, TextLine]]:
        out = []
        for ln in lines:
            if ln.text.strip() in _METRIC_NOISE:
                continue
            if parse_count(ln.text) is not None:
                out.append((ln.frame_center_y, ln))
        return out

    # 只认最右侧数据栏竖条（x>=0.90）里的数字。放宽到 0.70 会把视频字幕、
    # 《片名》标题、@昵称水印 误当成点赞评论数。
    right_full = [ln for ln in all_lines if ln.region == "full" and ln.frame_box[0] >= 0.90]
    full_nums = _numeric(right_full)
    if len(full_nums) >= 2:
        merged = right_full
    else:
        merged = sorted(rail_lines + right_full, key=lambda l: l.frame_center_y)

    ordered: List[TextLine] = []
    for ln in sorted(merged, key=lambda l: -l.score):
        txt = ln.text.strip()
        if not txt:
            continue
        if any(k.text.strip() == txt and abs(k.frame_center_y - ln.frame_center_y) < 0.03
               for k in ordered):
            continue
        ordered.append(ln)
    ordered.sort(key=lambda l: l.frame_center_y)
    m.raw_texts = [ln.text.strip() for ln in ordered]

    numbers: List[Tuple[float, int]] = []
    for ln in ordered:
        txt = ln.text.strip()
        if txt in _METRIC_NOISE:
            continue
        v = parse_count(txt)
        if v is not None and v >= 0:
            numbers.append((ln.frame_center_y, v))

    # 语义纠正：文本里明确写了"赞"
    for ln in merged:
        for mm in RE_LIKE_INLINE.finditer(ln.text):
            v = parse_count(mm.group(1) + mm.group(2))
            if v is not None:
                m.like = v

    # 抖音 PC 互动栏自上而下：点赞、评论、收藏、分享
    values = [v for _y, v in numbers]
    if m.like is None and len(values) >= 1:
        m.like = values[0]
    if len(values) >= 2:
        m.comment = values[1]
    if len(values) >= 3:
        m.collect = values[2]
    if len(values) >= 4:
        m.share = values[3]

    # 播放量：推荐流通常不显示，识别到就带上
    for ln in all_lines + rail_lines:
        mm = re.search(r"([\d]+(?:\.\d+)?[万wW千kK亿]?)\s*(?:次播放|播放)", ln.text)
        if mm:
            v = parse_count(mm.group(1))
            if v is not None:
                m.play = v
                break
    return m


# ------------------------------------------------------------------ 评论提取
def extract_comments(comment_lines: List[TextLine]) -> List[str]:
    out: List[str] = []
    for ln in sorted(comment_lines, key=lambda l: l.box[1]):
        t = ln.text.strip()
        if len(t) < 2 or len(t) > 60:
            continue
        if t in _METRIC_NOISE:
            continue
        if re.fullmatch(r"[\d\.,]+[万wW千kK亿]?", t):
            continue
        if re.search(r"^\d{1,2}[-/月]\d{1,2}|回复$|^展开\d+条回复$|^\d+条回复$", t):
            continue
        out.append(t)
    return out


def normalize_for_repeat(text: str) -> str:
    """评论去噪，用于重复度计算。"""
    return _SPAM_CHARS.sub("", text).strip().casefold()
