# -*- coding: utf-8 -*-
r"""OCR 区域标定工具。

用途：抖音客户端每个版本的界面布局略有差异，config.yaml 里的 ocr.regions
      （裁剪区域）可能需要按你的实际窗口微调。本工具会截取当前抖音窗口，
      把各区域框画出来、并叠加 10% 网格坐标，方便你读数值。

运行：用项目自带 Python 执行 tools\tune_regions.py
输出：data\preview\tune_regions.jpg
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.config import CONFIG, ROOT_DIR
from src.winutil import (Screen, enable_dpi_awareness, find_douyin_window,
                         grab_window, is_minimized)

COLORS = {
    "full": (90, 90, 90),
    "caption": (0, 200, 255),
    "rail": (60, 220, 60),
    "profile": (255, 170, 0),
    "comments": (200, 80, 220),
}


def main() -> int:
    enable_dpi_awareness()
    win = find_douyin_window(CONFIG.get("window.title_keywords", []), CONFIG.get("window.process_name", ""))
    if not win:
        print("未找到抖音窗口，请先打开抖音电脑客户端并让窗口可见。")
        return 2
    if is_minimized(win.hwnd):
        print("抖音窗口被最小化了，请先还原再运行本工具。")
        return 2

    screen = Screen()
    frame = grab_window(win, screen)
    screen.close()
    h, w = frame.shape[:2]
    print(f"已截取窗口 {win.title!r}  {w}x{h}")

    canvas = frame.copy()

    # 10% 网格 + 坐标刻度，方便直接读出百分比
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(canvas, (x, 0), (x, h), (110, 110, 110), 1)
        cv2.line(canvas, (0, y), (w, y), (110, 110, 110), 1)
        cv2.putText(canvas, f"{i / 10:.1f}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"{i / 10:.1f}", (4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    regions = CONFIG.get("ocr.regions", {})
    print("\n当前 config.yaml 的 ocr.regions：")
    for name, r in regions.items():
        try:
            l, t, rr, b = [float(v) for v in r]
        except Exception:
            continue
        x0, y0, x1, y1 = int(w * l), int(h * t), int(w * rr), int(h * b)
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, name, (x0 + 4, max(16, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"  {name:<10} [{l}, {t}, {rr}, {b}]   ->  像素 ({x0}, {y0}) ~ ({x1}, {y1})")

    # 右侧固定标尺：点赞按钮常见位置的参考线
    print("\n提示：自动点赞 coord 模式需要 relative_xy（点赞按钮中心的相对坐标）。")
    print("     在下面这张图里找到点赞（心形）图标中心，读出它的 x/10 和 y/10 即可。")

    outdir = os.path.join(ROOT_DIR, "data", "preview")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "tune_regions.jpg")
    cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(out)
    raw = os.path.join(outdir, "tune_regions_raw.jpg")
    cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(raw)

    print(f"\n已输出：\n  带网格标注：{out}\n  原始截图  ：{raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
