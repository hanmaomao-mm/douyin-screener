# -*- coding: utf-8 -*-
"""配置与关键词库的【热加载】管理。

工作流运行期间，随时修改 config.yaml / keywords.txt / spam_phrases.txt，
下一轮循环会自动读取最新内容，不需要重启。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------- 项目根目录
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# ---------------------------------------------------------------- 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "window": {
        "title_keywords": ["抖音", "Douyin"],
        "process_name": "douyin.exe",
        "keep_foreground": True,
        "restore_if_minimized": False,
        "auto_restore_hidden": True,
        "launch_if_not_running": True,
        "executable": "",
        "auto_switch_feed": True,
        "feed_tab": "推荐",
    },
    "rhythm": {
        "scroll_method": "key",
        "dwell_seconds": [2, 5],
        "scroll_steps": [2, 6],
        "scroll_segments": [1, 3],
        "micro_pause": [0.15, 0.7],
        "rest_every": [8, 15],
        "rest_seconds": [30, 90],
        "mouse_jitter": True,
        "scroll_settle": [1.2, 2.5],
    },
    "vision": {
        "enabled": True,
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "model": "deepseek-flash",
        "thinking": False,
        "json_mode": True,
        "timeout": 90,
        "max_tokens": 900,
        "temperature": 0.1,
        "retries": 2,
        "frames_per_video": 2,
        "prefilter": {
            "enabled": True,
            "strict": False,
            "min_face_px": 140,
            "min_eye_sharpness": 45,
            "min_face_ratio": 0.02,
            "max_face_ratio": 0.98,
            "face_model": "models/face_detection_yunet_2023mar.onnx",
            "auto_download_model": True,
        },
        "fallback_local": True,
    },
    "capture": {
        "sample_interval": [0.8, 1.4],
        "save_frame_on_hit": True,
        "save_frame_dir": "data/frames",
        "jpeg_quality": 88,
        "max_width": 1280,
    },
    "ocr": {
        "engine": "auto",
        "regions": {
            "full": [0.0, 0.0, 1.0, 1.0],
            "caption": [0.08, 0.70, 0.90, 0.925],
            "rail": [0.93, 0.30, 1.0, 0.85],
            "profile": [0.08, 0.70, 0.65, 0.83],
            "comments": [0.05, 0.04, 0.97, 0.55],
        },
        "min_confidence": 0.55,
        "upscale": 1.8,
        "upscale_by_region": {"full": 1.0, "caption": 2.0, "rail": 2.2,
                              "profile": 2.0, "comments": 1.3},
        "search_term_prefixes": ["看后搜", "大家都在搜", "大家在搜", "搜一搜", "搜索", "找同款"],
        "search_term_heuristics": True,
    },
    "keywords": {
        "file": "keywords.txt",
        "match_scope": ["search_terms", "hashtags"],
        "match_mode": "contains",
        "case_sensitive": False,
        "fuzzy": True,
    },
    # 美瞳内容自动识别（不依赖关键词库也能认出"在卖美瞳"的视频）
    "theme_detect": {
        "enabled": True,
        "use_ai": True,
        "counts_as_keyword_hit": True,
        "reject_unrelated": True,
        "words": ["美瞳", "彩瞳", "彩色隐形眼镜", "隐形眼镜", "瞳片", "彩片",
                  "美瞳推荐", "美瞳测评", "日抛", "月抛", "硅水凝胶", "瞳色"],
    },
    "filter": {
        "like_rate_enabled": True,
        "like_rate_threshold": 0.02,
        "play_count_mode": "auto",
        "manual_play_count": 0,
        "like_floor_enabled": True,
        "like_floor": 50,
        "like_floor_min_play": 10000,
        "comment_spam": {
            "enabled": True,
            "repeat_ratio": 0.4,
            "min_comments": 6,
            "spam_phrase_hit": 3,
            "phrases_file": "spam_phrases.txt",
        },
        "record_spam_to_excel": False,
        "record_rejected_to_excel": False,
    },
    "auto_like": {
        "enabled": False,
        "method": "template",
        "template_dir": "templates",
        "template_threshold": 0.86,
        "relative_xy": [0.918, 0.472],
        "delay_after_like": [3, 6],
        "max_likes_per_run": 30,
        "like_cooldown": [8, 20],
    },
    "grade": {
        "high_match_score": 75,
        "normal_score": 45,
        "weights": {"vision": 0.50, "keyword": 0.35, "metrics": 0.15},
    },
    "output": {
        "excel_path": "data/达人筛选结果.xlsx",
        "sheet_name": "候选达人",
        "log_dir": "logs",
        "save_full_log": False,
        "console_verbose": True,
        "excel_freeze_header": True,
    },
    "safety": {
        "dedup_memory": 400,
        "same_frame_retry": 3,
        "popup_on_lost_window": True,
        "max_runtime_minutes": 0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """把用户配置合并到默认配置上，缺失项自动回落到默认值。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _clean_lines(path: str) -> List[str]:
    """读取一个 txt，去掉注释行和空行，返回去重后的列表。"""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read().splitlines()
    except OSError:
        return []
    items, seen = [], set()
    for line in raw:
        text = line.strip()
        if not text or text.startswith("#") or text.startswith("//"):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


@dataclass
class KeywordLib:
    path: str
    words: List[str] = field(default_factory=list)
    mtime: float = 0.0
    loaded_at: float = 0.0

    @property
    def empty(self) -> bool:
        """关键词库为空 → 按需求：不做关键词限制。"""
        return len(self.words) == 0


@dataclass
class SpamPhrases:
    path: str
    phrases: List[str] = field(default_factory=list)
    mtime: float = 0.0


class HotConfig:
    """线程安全的配置容器，自动感知文件变更。"""

    def __init__(self, config_path: str | None = None):
        self.config_path = _abs(config_path or "config.yaml")
        self._lock = threading.RLock()
        self._cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self._cfg_mtime = -1.0
        self._kw = KeywordLib(path=_abs(DEFAULT_CONFIG["keywords"]["file"]))
        self._spam = SpamPhrases(path=_abs(DEFAULT_CONFIG["filter"]["comment_spam"]["phrases_file"]))
        self._reload_config(force=True)
        self._reload_keywords(force=True)
        self._reload_spam(force=True)

    # ------------------------------------------------ 内部加载
    def _reload_config(self, force: bool = False) -> bool:
        with self._lock:
            mt = _mtime(self.config_path)
            if not force and mt == self._cfg_mtime:
                return False
            data: Dict[str, Any] = {}
            if os.path.isfile(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8-sig") as f:
                        data = yaml.safe_load(f) or {}
                except Exception:
                    data = {}
            self._cfg = _deep_merge(DEFAULT_CONFIG, data)
            # 允许用环境变量注入 API Key，避免把密钥写进配置文件（也方便 CI）
            env_key = (os.environ.get("VISION_API_KEY")
                       or os.environ.get("DEEPSEEK_API_KEY")
                       or os.environ.get("DASHSCOPE_API_KEY"))
            if env_key:
                self._cfg.setdefault("vision", {})["api_key"] = env_key
            self._cfg_mtime = mt
            return True

    def _reload_keywords(self, force: bool = False) -> bool:
        with self._lock:
            path = _abs(self._cfg.get("keywords", {}).get("file", "keywords.txt"))
            mt = _mtime(path)
            if not force and path == self._kw.path and mt == self._kw.mtime:
                return False
            self._kw = KeywordLib(path=path, words=_clean_lines(path), mtime=mt, loaded_at=time.time())
            return True

    def _reload_spam(self, force: bool = False) -> bool:
        with self._lock:
            path = _abs(self._cfg.get("filter", {}).get("comment_spam", {}).get("phrases_file", "spam_phrases.txt"))
            mt = _mtime(path)
            if not force and path == self._spam.path and mt == self._spam.mtime:
                return False
            self._spam = SpamPhrases(path=path, phrases=_clean_lines(path), mtime=mt)
            return True

    # ------------------------------------------------ 对外接口
    def refresh(self) -> List[str]:
        """检测文件变化并热更新，返回发生变化的内容名列表。"""
        changed = []
        try:
            if self._reload_config():
                changed.append("config.yaml")
        except Exception:
            pass
        try:
            if self._reload_keywords():
                changed.append("关键词库")
        except Exception:
            pass
        try:
            if self._reload_spam():
                changed.append("水军话术库")
        except Exception:
            pass
        return changed

    @property
    def data(self) -> Dict[str, Any]:
        with self._lock:
            return self._cfg

    def get(self, dotted: str, default: Any = None) -> Any:
        """支持 'vision.model' 这种点号路径取值。"""
        with self._lock:
            cur: Any = self._cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def keywords(self) -> KeywordLib:
        with self._lock:
            return self._kw

    @property
    def spam_phrases(self) -> SpamPhrases:
        with self._lock:
            return self._spam

    # ------------------------------------------------ 路径工具
    def path(self, dotted: str, default: str = "") -> str:
        return _abs(self.get(dotted, default) or default)


CONFIG = HotConfig()
