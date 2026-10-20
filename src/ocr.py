# -*- coding: utf-8 -*-
"""画面文字识别（OCR）。

支持三种引擎，按可用性自动选择：
  - rapidocr : 本地中文 OCR，快、免费（首选）
  - vision   : 借用多模态大模型读字（无本地 OCR 时的兜底）
  - none     : 关闭文字识别

输出统一为 List[TextLine]，带文本、置信度、归一化坐标框。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from .config import CONFIG

# 屏蔽 RapidOCR / OpenCV 自己的 INFO / WARNING 刷屏，只保留真正的错误
for _name in ("RapidOCR", "rapidocr", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    cv2.setLogLevel(0)
except Exception:
    pass


@dataclass
class TextLine:
    text: str
    score: float
    box: Tuple[float, float, float, float]        # 相对所属裁剪区域的归一化框
    region: str = "full"
    frame_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # 相对整幅画面的归一化框
    upscale: float = 1.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def frame_center_y(self) -> float:
        return (self.frame_box[1] + self.frame_box[3]) / 2

    @property
    def frame_x(self) -> float:
        return self.frame_box[0]


# ------------------------------------------------------------------ 引擎探测
def _try_import_rapidocr():
    try:
        from rapidocr import RapidOCR  # rapidocr >= 3.x
        return RapidOCR, "new"
    except Exception:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR  # 旧包
        return RapidOCR, "old"
    except Exception:
        return None, "none"


_RAPID = None
_RAPID_KIND = "none"
_RAPID_TRIED = False


def _get_rapid():
    global _RAPID, _RAPID_KIND, _RAPID_TRIED
    if _RAPID_TRIED:
        return _RAPID
    _RAPID_TRIED = True
    cls, kind = _try_import_rapidocr()
    if cls is None:
        return None
    try:
        _RAPID = cls()
        _RAPID_KIND = kind
    except Exception as exc:  # 后端缺失（如未装 onnxruntime）
        print(f"[OCR] rapidocr 初始化失败，将自动降级：{exc}")
        _RAPID = None
    return _RAPID


def engine_status() -> str:
    eng = CONFIG.get("ocr.engine", "auto")
    if eng == "none":
        return "none"
    if eng in ("auto", "rapidocr") and _get_rapid() is not None:
        return f"rapidocr({_RAPID_KIND})"
    if eng == "vision":
        return "vision"
    return "vision" if _get_rapid() is None else "rapidocr"


# ------------------------------------------------------------------ 单张识别
def _run_rapid(img: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    eng = _get_rapid()
    if eng is None:
        return []
    try:
        res = eng(img)
    except Exception:
        return []

    out: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    # 新版返回 RapidOCRResult（带 .boxes/.txts/.scores），旧版返回 (list, elapse)
    txts = getattr(res, "txts", None)
    boxes = getattr(res, "boxes", None)
    scores = getattr(res, "scores", None)
    if txts is None and isinstance(res, tuple):
        payload = res[0]
        txts, boxes, scores = [], [], []
        for item in payload or []:
            try:
                box, text, score = item[0], item[1], item[2]
            except Exception:
                continue
            boxes.append(box)
            txts.append(text)
            scores.append(score)
    if txts is None:
        return []

    scores = list(scores) if scores is not None else [1.0] * len(txts)
    boxes = list(boxes) if boxes is not None else [None] * len(txts)
    for text, score, box in zip(txts, scores, boxes):
        if not text:
            continue
        if box is None:
            bbox = (0.0, 0.0, 1.0, 1.0)
        else:
            try:
                arr = np.asarray(box, dtype=float).reshape(-1, 2)
                bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                        float(arr[:, 0].max()), float(arr[:, 1].max()))
            except Exception:
                bbox = (0.0, 0.0, 1.0, 1.0)
        out.append((str(text).strip(), float(score), bbox))
    return out


def ocr_region(img: np.ndarray, region: str, min_conf: float = 0.55,
               upscale: float = 1.0,
               offset: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
               frame_size: Tuple[int, int] = (0, 0)) -> List[TextLine]:
    """识别一块裁剪图（BGR ndarray）。

    offset / frame_size 用于把区域内的坐标换算回整幅画面坐标（frame_box），
    这样不同区域识别出来的文字可以放在同一套坐标系里比较。
    """
    if img is None or img.size == 0:
        return []

    scaled = img
    if upscale and upscale > 1.01:
        scaled = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    h, w = scaled.shape[:2]
    raw = _run_rapid(scaled)
    lines: List[TextLine] = []

    fw, fh = frame_size
    ox, oy, ow, oh = offset  # 区域在整幅画面中的归一化位置与大小
    for text, score, (x0, y0, x1, y1) in raw:
        if score < min_conf and score > 0:
            continue
        nb = (x0 / max(w, 1), y0 / max(h, 1), x1 / max(w, 1), y1 / max(h, 1))
        if fw and fh:
            fb = (ox + nb[0] * ow, oy + nb[1] * oh, ox + nb[2] * ow, oy + nb[3] * oh)
        else:
            fb = nb
        lines.append(TextLine(text=text, score=score, box=nb, region=region,
                              frame_box=fb, upscale=upscale))
    return lines


def ocr_regions(frame: np.ndarray, regions: Dict[str, List[float]], min_conf: float = 0.55,
                upscale: float = 1.0,
                upscale_by_region: Optional[Dict[str, float]] = None) -> Dict[str, List[TextLine]]:
    """按配置的区域切块识别。frame 是整窗截图（BGR）。

    upscale_by_region 可以给每个区域单独指定放大倍数：
      · full 这种大图应该用 1.0（放大会让 OCR 变成几倍耗时，收益却很小）
      · rail / caption 这类小字区域放大 1.6~2.2，识别率提升明显
    """
    h, w = frame.shape[:2]
    per = upscale_by_region or {}
    result: Dict[str, List[TextLine]] = {}
    for name, region in (regions or {}).items():
        try:
            l, t, r, b = [float(v) for v in region]
        except Exception:
            continue
        x0, y0 = int(w * l), int(h * t)
        x1, y1 = int(w * r), int(h * b)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            result[name] = []
            continue
        scale = float(per.get(name, upscale))
        result[name] = ocr_region(crop, name, min_conf, scale,
                                  offset=(l, t, r - l, b - t), frame_size=(w, h))
    return result


def all_text(lines_by_region: Dict[str, List[TextLine]]) -> str:
    return " ".join(ln.text for group in lines_by_region.values() for ln in group)


def ocr_frame(frame: np.ndarray) -> Dict[str, List[TextLine]]:
    """按当前配置识别整帧（自动读取区域 / 置信度 / 分区域放大倍数）。"""
    return ocr_regions(
        frame,
        CONFIG.get("ocr.regions", {}),
        float(CONFIG.get("ocr.min_confidence", 0.55)),
        float(CONFIG.get("ocr.upscale", 1.0)),
        CONFIG.get("ocr.upscale_by_region", {}),
    )
