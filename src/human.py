# -*- coding: utf-8 -*-
"""拟人化操作：慢速滚动、随机停顿、休息、自动点赞。

设计原则（防风控）：
  1. 所有动作间隔都带随机扰动，绝不定频
  2. 一次下滑拆成几小段小步滚动，模仿真实滚轮的顿挫感
  3. 每次滚动前鼠标会轻微移动一下，避免鼠标长时间悬停在同一点
  4. 连续刷 N 条后强制休息 30~90 秒

【本模块只允许三种动作】：鼠标移动、滚轮滚动、点击点赞按钮。
   绝不包含任何评论 / 私信 / 关注 / 分享 的自动化。
"""
from __future__ import annotations

import os
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui

from .config import CONFIG
from .winutil import grab_window, WinInfo

# 紧急制动：把鼠标猛甩到屏幕左上角 (0,0) 即可立即中断脚本（pyautogui 内置 FAILSAFE）
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def rnd(rng, lo: float = None, hi: float = None) -> float:
    """从 [lo, hi] 或二元列表里取随机值。"""
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        lo, hi = float(rng[0]), float(rng[1])
    lo = 0.0 if lo is None else float(lo)
    hi = lo if hi is None else float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def rnd_int(rng, lo: int = None, hi: int = None) -> int:
    return int(round(rnd(rng, lo, hi)))


class Human:
    """对抖音窗口的一系列拟人化操作。"""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._video_count = 0
        self._next_rest_at = rnd_int(CONFIG.get("rhythm.rest_every", [8, 15]))
        self._like_count = 0
        self._last_like_ts = 0.0
        self._templates: Optional[dict] = None

    # ------------------------------------------------------------ 基础动作
    def micro_pause(self) -> None:
        time.sleep(rnd(CONFIG.get("rhythm.micro_pause", [0.15, 0.7])))

    def move_to(self, x: int, y: int, jitter: int = 6) -> None:
        x = int(max(1, min(self.screen_w - 2, x + random.randint(-jitter, jitter))))
        y = int(max(1, min(self.screen_h - 2, y + random.randint(-jitter, jitter))))
        pyautogui.moveTo(x, y, duration=random.uniform(0.15, 0.45), tween=pyautogui.easeInOutQuad)

    def _content_point(self, win: WinInfo) -> Tuple[int, int]:
        """取视频内容区的一个随机点（避开左右边栏）。"""
        l, t = win.rect[0], win.rect[1]
        x = l + int(win.width * random.uniform(0.28, 0.60))
        y = t + int(win.height * random.uniform(0.30, 0.70))
        return x, y

    # ------------------------------------------------------------ 滚动
    def scroll_next(self, win: WinInfo, settle: bool = True, method: str = None) -> None:
        """切到下一条视频。

        实测抖音 PC 客户端【滚轮滚动无效】，方向键 ↓ 才能切换视频，
        所以默认用按键；wheel 模式保留给特殊情况兜底。
        """
        rhythm = CONFIG.get("rhythm", {})
        m = (method or str(rhythm.get("scroll_method", "key"))).lower()

        x, y = self._content_point(win)
        if CONFIG.get("rhythm.mouse_jitter", True):
            self.move_to(x, y)
        else:
            pyautogui.moveTo(x, y, duration=0.1)
        self.micro_pause()

        if m in ("key", "auto"):
            pyautogui.press("down")
        if m in ("wheel", "auto"):
            total = rnd_int(rhythm.get("scroll_steps", [2, 6]))
            segments = max(1, rnd_int(rhythm.get("scroll_segments", [1, 3])))
            remaining = total
            for i in range(segments):
                part = remaining if i == segments - 1 else max(1, remaining // (segments - i) + random.randint(-1, 1))
                part = max(1, min(part, remaining))
                remaining -= part
                pyautogui.scroll(-part)
                if remaining > 0:
                    time.sleep(random.uniform(0.08, 0.28))
        if settle:
            time.sleep(rnd(rhythm.get("scroll_settle", [1.2, 2.5])))

    def maybe_rest(self) -> bool:
        """到点就休息，返回是否休息过。"""
        self._video_count += 1
        if self._video_count < self._next_rest_at:
            return False
        secs = rnd(CONFIG.get("rhythm.rest_seconds", [30, 90]))
        print(f"  😴 连续浏览 {self._video_count} 条，休息 {secs:.0f} 秒（模拟真人）...")
        self._video_count = 0
        self._next_rest_at = rnd_int(CONFIG.get("rhythm.rest_every", [8, 15]))
        deadline = time.time() + secs
        while time.time() < deadline:
            time.sleep(min(5.0, deadline - time.time()))
            if self._screen_locked():
                continue
        return True

    @staticmethod
    def _screen_locked() -> bool:
        """粗略检测锁屏（用户按 Win+L），锁屏期间不动作。"""
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetForegroundWindow() == 0)
        except Exception:
            return False

    # ------------------------------------------------------------ 一次性导航（切频道）
    def click_text(self, win: WinInfo, screen, targets: List[str],
                   dry_run: bool = False) -> Tuple[bool, str]:
        """在画面里 OCR 找到指定文字并模拟点击（用于启动后切到「推荐」页等一次性导航）。

        优先点完全相等的文字，其次点包含目标的短文本；同等条件选最靠左的
        （频道导航在左侧栏，避免误点视频文案里的同名词）。
        """
        from . import ocr as ocr_mod  # 延迟导入，避免循环依赖

        frame = grab_window(win, screen)
        lines = ocr_mod.ocr_region(frame, "nav", 0.5, 1.0)
        best_key, best_ln = None, None
        for ln in lines:
            t = ln.text.strip()
            for tg in targets:
                if t == tg:
                    key = (2, -ln.frame_x, ln.score)
                elif tg in t and len(t) <= len(tg) + 2:
                    key = (1, -ln.frame_x, ln.score)
                else:
                    continue
                if best_key is None or key > best_key:
                    best_key, best_ln = key, ln

        if best_ln is None:
            return False, "画面中未找到「" + "/".join(targets) + "」入口"

        ln = best_ln
        fx = (ln.frame_box[0] + ln.frame_box[2]) / 2
        fy = (ln.frame_box[1] + ln.frame_box[3]) / 2
        if dry_run:
            return True, f"[dry-run] 命中「{ln.text.strip()}」@ ({fx:.2f},{fy:.2f}) 相似度{ln.score:.2f}"

        px = win.rect[0] + int(win.width * fx)
        py = win.rect[1] + int(win.height * fy)
        self.move_to(px, py, jitter=3)
        time.sleep(random.uniform(0.2, 0.5))
        pyautogui.click()
        time.sleep(random.uniform(1.2, 2.0))   # 等页面切换
        return True, f"已点击「{ln.text.strip()}」"

    # ------------------------------------------------------------ 自动点赞
    def _load_templates(self) -> dict:
        if self._templates is not None:
            return self._templates
        cfg = CONFIG.get("auto_like", {})
        tdir = CONFIG.path("auto_like.template_dir", "templates")
        out = {}
        for name in ("like_normal", "like_active"):
            p = os.path.join(tdir, f"{name}.png")
            if os.path.isfile(p):
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    out[name] = img
        self._templates = out
        return out

    def _find_like_button(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """在画面右侧栏做模板匹配，返回 (相对x, 相对y, 相似度)。"""
        cfg = CONFIG.get("auto_like", {})
        if str(cfg.get("method", "template")).lower() != "template":
            return None
        tpls = self._load_templates()
        tpl = tpls.get("like_normal") or tpls.get("like_active")
        if tpl is None:
            return None

        h, w = frame.shape[:2]
        # 只在右侧 30% 区域内找，提速并避免误匹配
        x0 = int(w * 0.70)
        roi = frame[:, x0:]
        th, tw = tpl.shape[:2]
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            return None
        try:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None
        _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
        if maxv < float(cfg.get("template_threshold", 0.86)):
            return None
        cx = x0 + maxl[0] + tw // 2
        cy = maxl[1] + th // 2
        return cx / w, cy / h, float(maxv)

    def can_like(self) -> bool:
        cfg = CONFIG.get("auto_like", {})
        if not cfg.get("enabled", False):
            return False
        if self._like_count >= int(cfg.get("max_likes_per_run", 30)):
            return False
        if time.time() - self._last_like_ts < rnd(cfg.get("like_cooldown", [8, 20])):
            return False
        return True

    def click_like(self, win: WinInfo, frame: np.ndarray) -> Tuple[bool, str]:
        """点击点赞按钮。返回 (是否成功, 说明)。"""
        cfg = CONFIG.get("auto_like", {})
        method = str(cfg.get("method", "template")).lower()
        rel = None
        note = ""

        if method == "template":
            found = self._find_like_button(frame)
            if found:
                rel = (found[0], found[1])
                note = f"模板匹配 相似度{found[2]:.2f}"
            elif self._load_templates():
                return False, "模板未匹配到点赞按钮，本次跳过点赞"
            else:
                note = "未放模板图，回退坐标模式"

        if rel is None:
            rel = tuple(cfg.get("relative_xy", [0.918, 0.472]))
            note = note or "相对坐标模式"

        px = win.rect[0] + int(win.width * float(rel[0]))
        py = win.rect[1] + int(win.height * float(rel[1]))
        self.micro_pause()
        self.move_to(px, py, jitter=4)
        time.sleep(random.uniform(0.12, 0.35))
        pyautogui.click()
        self._like_count += 1
        self._last_like_ts = time.time()
        return True, note

    @property
    def like_count(self) -> int:
        return self._like_count
