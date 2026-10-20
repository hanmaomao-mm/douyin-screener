# -*- coding: utf-8 -*-
"""日志：控制台 + 按天文件 + JSONL 全量决策流水。"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

_LOCK = threading.Lock()
_JSONL_PATH: str | None = None


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def setup_logger(log_dir: str, verbose: bool = True) -> logging.Logger:
    """初始化主日志器。"""
    global _JSONL_PATH
    _ensure_dir(log_dir)
    day = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"run_{day}.log")
    _JSONL_PATH = os.path.join(log_dir, f"decisions_{day}.jsonl")

    logger = logging.getLogger("douyin_screener")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    fh = RotatingFileHandler(log_file, maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def write_jsonl(record: dict, enabled: bool = True) -> None:
    """把每一条视频的完整决策写入 JSONL（含被过滤掉的），便于事后复盘。"""
    if not enabled or not _JSONL_PATH:
        return
    record = dict(record)
    record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    try:
        with _LOCK:
            with open(_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def banner(text: str, width: int = 74) -> str:
    line = "=" * width
    return f"\n{line}\n{text}\n{line}"
