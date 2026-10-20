# -*- coding: utf-8 -*-
"""判定引擎：关键词匹配 + 刷量号识别 + 等级评分。

判定顺序：
  1. 广告 / 直播  → 直接跳过
  2. 视觉不合适（无真人 / 遮挡 / 眼睛看不清） → 记为「不达标」
  3. 关键词库非空时，未命中任何关键词 → 记为「不达标」
  4. 刷量号规则 → 标记「刷量号」
  5. 其余按综合分给出「高匹配 / 普通」
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import CONFIG, KeywordLib, SpamPhrases
from .parser import Metrics, normalize_for_repeat
from .vision import VisionResult

GRADE_HIGH = "高匹配"
GRADE_NORMAL = "普通"
GRADE_SPAM = "刷量号"
GRADE_REJECT = "不达标"

# 广告 / 直播 画面特征词
_AD_WORDS = ["广告", "推广", "赞助", "商品", "购物车", "小黄车", "立即购买", "去看看", "直播间", "正在直播", "直播中"]


# ------------------------------------------------------------------ 数据结构
@dataclass
class Decision:
    grade: str = GRADE_REJECT
    score: float = 0.0
    hit_keywords: List[str] = field(default_factory=list)
    spam_reasons: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)
    is_spam: bool = False
    is_candidate: bool = False
    like_rate: Optional[float] = None
    like_rate_note: str = ""
    reason_text: str = ""
    metrics_score: float = 0.0
    keyword_score: float = 0.0
    vision_score: float = 0.0


# ------------------------------------------------------------------ 关键词匹配
def _levenshtein_le1(a: str, b: str) -> bool:
    """编辑距离 <= 1 的快速判断（应对 OCR 少量错字）。"""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return diff <= 1
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] != b[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


def match_keywords(lib: KeywordLib,
                   search_terms: List[str],
                   hashtags: List[str],
                   scope: List[str],
                   mode: str = "contains",
                   case_sensitive: bool = False,
                   fuzzy: bool = True) -> List[str]:
    """返回命中的关键词列表。关键词库为空 → 返回空列表（调用方据此跳过该规则）。"""
    if lib.empty:
        return []

    targets: List[Tuple[str, str]] = []
    if "search_terms" in scope:
        targets += [("看后搜", t) for t in search_terms]
    if "hashtags" in scope:
        targets += [("话题", t) for t in hashtags]
    if not targets:
        return []

    hits: List[str] = []
    for word in lib.words:
        w = word if case_sensitive else word.casefold()
        for _kind, target in targets:
            t = target if case_sensitive else target.casefold()
            ok = False
            if mode == "exact":
                ok = (t == w)
            else:
                ok = (w in t)
            if not ok and fuzzy and len(w) >= 3:
                # 滑窗比对，容忍 OCR 错字
                for i in range(0, max(1, len(t) - len(w) + 1)):
                    if _levenshtein_le1(t[i:i + len(w)], w):
                        ok = True
                        break
            if ok:
                if word not in hits:
                    hits.append(word)
                break
    return hits


# ------------------------------------------------------------------ 美瞳内容识别
def detect_theme(vision: VisionResult, texts: List[str], cfg: Dict[str, Any]) -> List[str]:
    """判断这条视频是否属于「美瞳相关内容」，返回命中的主题词。

    两个来源：
      1) AI 视觉（vision.source == "api"）：模型直接读画面文字/贴纸给出的结论
      2) 本地词表：在【文案 + 看后搜 + 话题标签】里做包含匹配（兜底，API 挂了也能用）
    """
    if not cfg.get("enabled", True):
        return []
    hits: List[str] = []

    if cfg.get("use_ai", True) and vision.source == "api":
        for w in (vision.lens_brand_hits or []):
            w = str(w).strip()
            if w and w not in hits:
                hits.append(w)
        if vision.lens_content and not hits:
            hits.append("美瞳内容(AI识别)")

    for raw in (cfg.get("words") or []):
        w = str(raw).strip()
        if not w:
            continue
        wl = w.casefold()
        for t in texts:
            if wl in str(t or "").casefold():
                if w not in hits:
                    hits.append(w)
                break
    return hits


# ------------------------------------------------------------------ 刷量判定
def detect_comment_spam(comments: List[str], phrases: SpamPhrases) -> Tuple[bool, List[str]]:
    cfg = CONFIG.get("filter.comment_spam", {})
    reasons: List[str] = []
    if not cfg.get("enabled", True):
        return False, reasons
    if len(comments) < int(cfg.get("min_comments", 6)):
        return False, reasons

    norm = [normalize_for_repeat(c) for c in comments]
    norm = [n for n in norm if len(n) >= 2]
    if not norm:
        return False, reasons

    counter = Counter(norm)
    top_text, top_n = counter.most_common(1)[0]
    repeat_ratio = top_n / float(len(norm))
    if repeat_ratio >= float(cfg.get("repeat_ratio", 0.4)):
        reasons.append(f"评论高度重复（\"{top_text[:10]}\"出现{top_n}/{len(norm)}次，占比{repeat_ratio:.0%}）")

    phrase_set = [p.casefold() for p in (phrases.phrases or [])]
    hit_n = sum(1 for n in norm if any(p and p in n for p in phrase_set))
    if hit_n >= int(cfg.get("spam_phrase_hit", 3)):
        reasons.append(f"命中无意义话术库 {hit_n} 条")

    return len(reasons) > 0, reasons


def compute_like_rate(metrics: Metrics) -> Tuple[Optional[float], str]:
    """计算点赞率。抖音 PC 推荐流通常不展示播放量，此时按配置决定口径。"""
    cfg = CONFIG.get("filter", {})
    mode = str(cfg.get("play_count_mode", "auto")).lower()

    if not cfg.get("like_rate_enabled", True) or mode == "off":
        return None, "点赞率规则已关闭"
    if metrics.like is None:
        return None, "画面未识别到点赞数"

    play = metrics.play
    if mode == "manual":
        play = int(cfg.get("manual_play_count", 0) or 0)
        if play <= 0:
            return None, "manual 模式但未设置 manual_play_count"
        note = f"按手动基准播放量 {play} 估算"
    elif mode == "auto":
        if not play or play <= 0:
            return None, "画面未显示播放量（推荐流常见），该规则本条跳过"
        note = f"画面读取播放量 {play}"
    else:
        return None, f"未知 play_count_mode={mode}"

    rate = float(metrics.like) / float(play)
    return rate, note


def detect_spam(metrics: Metrics, comments: List[str], phrases: SpamPhrases) -> Tuple[bool, List[str], Optional[float], str]:
    cfg = CONFIG.get("filter", {})
    reasons: List[str] = []

    rate, note = compute_like_rate(metrics)
    if rate is not None and cfg.get("like_rate_enabled", True):
        thr = float(cfg.get("like_rate_threshold", 0.02))
        if rate < thr:
            reasons.append(f"点赞率 {rate:.2%} 低于阈值 {thr:.2%}（{note}）")

    if cfg.get("like_floor_enabled", True):
        floor = int(cfg.get("like_floor", 50))
        min_play = int(cfg.get("like_floor_min_play", 10000))
        if metrics.like is not None and metrics.like < floor:
            if metrics.play is None or metrics.play >= min_play:
                reasons.append(f"点赞量仅 {metrics.like}，低于下限 {floor}")

    spam_c, comment_reasons = detect_comment_spam(comments, phrases)
    reasons += comment_reasons
    return len(reasons) > 0, reasons, rate, note


# ------------------------------------------------------------------ 等级评分
def _metrics_health(metrics: Metrics) -> float:
    """数据健康度 0~100：点赞率高、互动均衡则分数高。"""
    if metrics.like is None:
        return 60.0
    score = 60.0
    rate = metrics.like_rate
    if rate is not None:
        score = min(100.0, 40.0 + rate * 3000.0)   # 3.3% 点赞率即满分
    if metrics.comment and metrics.like:
        cr = metrics.comment / float(metrics.like)
        if 0.005 <= cr <= 0.15:                    # 评论/点赞比处于正常区间
            score = min(100.0, score + 8.0)
        elif cr > 0.5:
            score = max(0.0, score - 20.0)         # 评论异常多，疑似刷评
    return max(0.0, min(100.0, score))


def judge(vision: VisionResult,
          search_confirmed: List[str],
          search_guessed: List[str],
          hashtags: List[str],
          metrics: Metrics,
          comments: List[str],
          all_text: str = "") -> Decision:
    cfg = CONFIG.get("filter", {})
    grade_cfg = CONFIG.get("grade", {})
    kw_cfg = CONFIG.get("keywords", {})
    lib = CONFIG.keywords
    phrases = CONFIG.spam_phrases

    d = Decision()

    # ---- 1. 广告 / 直播 ----
    if vision.is_ad_or_live or any(w in all_text for w in _AD_WORDS):
        d.grade = GRADE_REJECT
        d.reject_reasons.append("疑似广告或直播画面")
        d.reason_text = "疑似广告/直播，直接跳过"
        return d

    # ---- 2. 视觉门槛 ----
    vision_score = max(0.0, min(100.0, vision.lens_suitable))
    d.vision_score = vision_score
    if not vision.has_person:
        d.reject_reasons.append("画面无真人出镜")
    if not vision.face_visible:
        d.reject_reasons.append("面部被遮挡或不可见")
    if not vision.eyes_visible:
        d.reject_reasons.append("眼睛看不清，无法判断美瞳佩戴效果")
    if d.reject_reasons:
        d.reason_text = "；".join(d.reject_reasons)
        d.grade = GRADE_REJECT
        return d

    # ---- 3. 关键词 / 美瞳主题 ----
    scope = kw_cfg.get("match_scope", ["search_terms", "hashtags"])
    targets = list(search_confirmed) + list(hashtags)
    kw_hits = match_keywords(
        lib, targets, hashtags, scope,
        str(kw_cfg.get("match_mode", "contains")),
        bool(kw_cfg.get("case_sensitive", False)),
        bool(kw_cfg.get("fuzzy", True)),
    )

    # 美瞳内容识别（AI 视觉结论 + 本地主题词表）
    theme_cfg = CONFIG.get("theme_detect", {}) or {}
    scan_texts = list(targets) + ([all_text] if all_text else [])
    theme_hits = detect_theme(vision, scan_texts, theme_cfg)
    counts_as_hit = bool(theme_cfg.get("counts_as_keyword_hit", True))
    ai_ran = (vision.source == "api")   # API 没跑起来时，不能拿"识不出美瞳"去否决视频

    if kw_hits:
        d.hit_keywords = kw_hits
        d.keyword_score = 100.0
        kw_note = f"命中关键词：{'、'.join(kw_hits)}"
    elif theme_hits and counts_as_hit:
        d.hit_keywords = theme_hits
        d.keyword_score = 92.0
        kw_note = f"识别为美瞳内容：{'、'.join(theme_hits[:5])}"
    elif lib.empty:
        if theme_cfg.get("enabled", True) and theme_cfg.get("reject_unrelated", True) and ai_ran:
            d.grade = GRADE_REJECT
            d.reject_reasons.append("不是美瞳相关内容")
            d.reason_text = "画面与标签都不是美瞳内容，跳过"
            d.keyword_score = 0.0
            return d
        d.keyword_score = 70.0          # 关键词库为空 → 不做限制，给中性分
        kw_note = "关键词库为空，本维度不限制"
    else:
        d.grade = GRADE_REJECT
        d.reject_reasons.append("未命中关键词库中任何词汇")
        d.reason_text = "未命中关键词（看后搜与话题标签均无匹配）"
        d.keyword_score = 0.0
        return d

    # ---- 4. 刷量判定 ----
    is_spam, spam_reasons, rate, note = detect_spam(metrics, comments, phrases)
    d.like_rate, d.like_rate_note = rate, note
    d.metrics_score = _metrics_health(metrics)

    if is_spam:
        d.is_spam = True
        d.spam_reasons = spam_reasons
        d.grade = GRADE_SPAM
        d.reason_text = "刷量判定：" + "；".join(spam_reasons)
        d.score = round(0.5 * vision_score + 0.5 * d.metrics_score, 1)
        return d

    # ---- 5. 综合分 ----
    w = grade_cfg.get("weights", {}) or {}
    wv = float(w.get("vision", 0.5))
    wk = float(w.get("keyword", 0.35))
    wm = float(w.get("metrics", 0.15))
    total_w = wv + wk + wm or 1.0
    d.score = round((vision_score * wv + d.keyword_score * wk + d.metrics_score * wm) / total_w, 1)

    if d.score >= float(grade_cfg.get("high_match_score", 75)):
        d.grade = GRADE_HIGH
    elif d.score >= float(grade_cfg.get("normal_score", 45)):
        d.grade = GRADE_NORMAL
    else:
        d.grade = GRADE_REJECT
        d.reject_reasons.append(f"综合分 {d.score} 未达普通线")
        d.reason_text = f"综合分不足（{d.score}）"
        return d

    d.is_candidate = True
    parts = [vision.reason or f"眼部条件达标（{vision.eye_area_quality:.0f}分）", kw_note]
    if rate is not None:
        parts.append(f"点赞率 {rate:.2%}")
    else:
        parts.append(f"点赞率未计算（{note}）")
    if vision.content_type:
        parts.append(f"内容类型：{vision.content_type}")
    d.reason_text = " | ".join(p for p in parts if p)
    return d
