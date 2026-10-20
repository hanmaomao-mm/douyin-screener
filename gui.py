# -*- coding: utf-8 -*-
"""抖音美瞳达人筛选工作流 —— 可视化控制台（PySide6）。

替代黑框控制台：实时画面预览、逐条识别结果、日志、候选达人表格、
关键词库与点赞率阈值都直接在界面上改，改完即生效。
"""
from __future__ import annotations

import os
import re
import sys
import time
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# pythonw 启动时 stdout/stderr 为 None，任何 print 都会直接崩 —— 先兜底重定向
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
if sys.stdout is None:
    sys.stdout = open(os.path.join(BASE_DIR, "logs", "gui_stdout.log"), "a",
                      buffering=1, encoding="utf-8")
if sys.stderr is None:
    sys.stderr = sys.stdout

import cv2
import numpy as np
import pyautogui

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from main import (build_links, fmt_num, frame_fingerprint, hamming_hex,
                  is_bad_frame, process_one, startup_checks)
from src.config import CONFIG, ROOT_DIR
from src.excel_out import ExcelWriter
from src.human import Human, rnd
from src.judge import GRADE_HIGH, GRADE_NORMAL, GRADE_REJECT, GRADE_SPAM
from src.logsetup import setup_logger
from src.winutil import (Screen, activate, enable_dpi_awareness,
                         find_douyin_exe, grab_window, hide_window, is_alive,
                         is_minimized, launch_douyin, refresh_rect,
                         resolve_douyin_window, show_no_activate, window_status,
                         WinInfo)

# ---------------------------------------------------------------------------
GRADE_COLOR = {
    GRADE_HIGH: "#E5484D",     # 红：高匹配
    GRADE_NORMAL: "#F5A623",   # 橙：普通
    GRADE_SPAM: "#30A46C",     # 绿：刷量号
    GRADE_REJECT: "#6E7681",   # 灰：不达标
}

# 视觉服务商预设：(下拉框名称, provider 标识, base_url, model, Key 输入框提示)
PROVIDER_PRESETS = [
    ("DeepSeek（推荐，便宜）", "deepseek", "https://api.deepseek.com", "deepseek-flash",
     "sk-...（DeepSeek Key —— platform.deepseek.com 申请）"),
    ("阿里云百炼 qwen-vl", "dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "qwen-vl-max", "sk-...（阿里云百炼 Key —— bailian.console.aliyun.com 申请）"),
    ("自定义（自己填 base_url / model）", "custom", "", "",
     "sk-...（任意 OpenAI 兼容多模态接口的 Key）"),
]

QSS = """
QMainWindow, QDialog { background: #15171B; }
QWidget { color: #E2E5EA; font-size: 12px; }
QGroupBox {
    background: #1C1F25; border: 1px solid #2A2F38; border-radius: 8px;
    margin-top: 14px; padding: 10px 10px 10px 10px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #8B93A1; }
QPushButton {
    background: #262B33; border: 1px solid #30363F; border-radius: 6px;
    padding: 7px 14px; color: #E2E5EA;
}
QPushButton:hover { background: #2F3540; }
QPushButton:pressed { background: #232830; }
QPushButton:disabled { color: #565E6A; background: #1E2228; border-color: #262B33; }
QPushButton#startBtn { background: #2B7A4B; border-color: #2B7A4B; font-weight: 700; padding: 8px 26px; }
QPushButton#startBtn:hover { background: #339159; }
QPushButton#stopBtn  { background: #6E3238; border-color: #6E3238; font-weight: 700; }
QPushButton#stopBtn:hover { background: #833C44; }
QPlainTextEdit, QTextEdit {
    background: #16181C; border: 1px solid #2A2F38; border-radius: 6px;
    font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 12px;
}
QTableWidget {
    background: #16181C; border: 1px solid #2A2F38; border-radius: 6px;
    gridline-color: #242931; alternate-background-color: #191C21;
    selection-background-color: #2B4C77;
}
QHeaderView::section {
    background: #20242B; color: #9AA3B0; border: none;
    border-bottom: 1px solid #2A2F38; padding: 5px 8px; font-weight: 600;
}
QDoubleSpinBox { background: #16181C; border: 1px solid #2A2F38; border-radius: 5px; padding: 4px 6px; }
QTabWidget::pane { border: 1px solid #2A2F38; border-radius: 6px; background: #16181C; }
QTabBar::tab {
    background: transparent; color: #8B93A1; padding: 7px 18px;
    border: none; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #E2E5EA; border-bottom: 2px solid #4C8DFF; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #3A414C; border-radius: 4px; background: #16181C; }
QCheckBox::indicator:checked { background: #2B7A4B; border-color: #2B7A4B; }
QLabel#statValue { font-size: 17px; font-weight: 700; }
QLabel#hint { color: #737C89; font-size: 11px; }
"""


class _QtLogHandler(logging.Handler):
    """把 logging 日志桥接到界面日志面板。"""

    def __init__(self, emit):
        super().__init__()
        self._emit = emit

    def emit(self, record):
        try:
            self._emit(record.getMessage(), record.levelname.lower())
        except Exception:
            pass


def _update_yaml_value(dotted: str, value) -> bool:
    """原位更新 config.yaml 里的一个配置项（保留注释与其他内容）。"""
    section, leaf = dotted.split(".", 1)
    path = CONFIG.config_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return False
    in_sec = False
    leaf_pat = re.compile(r"^(\s*)" + re.escape(leaf) + r"\s*:")
    sec_pat = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(?:#.*)?$")
    done = False
    for i, ln in enumerate(lines):
        m = sec_pat.match(ln)
        if m:
            in_sec = (m.group(1) == section)
            continue
        if in_sec and leaf_pat.match(ln):
            indent = leaf_pat.match(ln).group(1)
            hash_idx = ln.find("#")
            comment = ("    " + ln[hash_idx:].rstrip()) if hash_idx >= 0 else ""
            if isinstance(value, bool):
                value = "true" if value else "false"
            lines[i] = f"{indent}{leaf}: {value}{comment}\n"
            done = True
            break
    if not done:
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except OSError:
        return False
    CONFIG.refresh()
    return True


class ApiTester(QThread):
    """后台测一下视觉 API 连通性，不卡界面。"""
    done = Signal(bool, str)

    def run(self):
        try:
            import requests
            cfg = CONFIG.get("vision", {})
            key = (cfg.get("api_key") or "").strip()
            base = str(cfg.get("base_url", "")).rstrip("/")
            model = str(cfg.get("model", ""))
            if not key:
                self.done.emit(False, "未填写 API Key —— 当前只能用本地降级，判断不了颜值")
                return
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
            if "deepseek" in base.lower() or str(cfg.get("provider", "")).lower() == "deepseek":
                body["thinking"] = {"type": "disabled"}    # 测试时不走思考模式，快一点
            r = requests.post(
                base + "/chat/completions",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=20,
            )
            if r.status_code == 200:
                self.done.emit(True, f"连接成功：{model} @ {base}")
            else:
                self.done.emit(False, f"HTTP {r.status_code}: {r.text[:160]}")
        except Exception as exc:                                   # noqa: BLE001
            self.done.emit(False, str(exc)[:160])


# ===========================================================================
# 后台工作线程
# ===========================================================================
class Worker(QThread):
    frameReady = Signal(object)        # BGR ndarray（当前视频画面）
    logLine = Signal(str, str)         # 文本, 级别(info/warn/error)
    resultReady = Signal(dict)         # 单条视频识别结果
    statsChanged = Signal(dict)
    stateChanged = Signal(str)         # starting / running / paused / stopped / error / idle

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self._pause = False
        self._started_at = 0.0
        self.stats = {"idx": 0, "high": 0, "normal": 0, "spam": 0,
                      "reject": 0, "liked": 0, "rows": 0}

    # ---------------- 外部控制 ----------------
    def request_stop(self):
        self._stop = True

    def set_pause(self, on: bool):
        self._pause = bool(on)

    # ---------------- 内部工具 ----------------
    def log(self, msg: str, level: str = "info"):
        self.logLine.emit(str(msg), level)

    def _make_logger(self) -> logging.Logger:
        logger = setup_logger(CONFIG.path("output.log_dir", "logs"), False)
        logger.addHandler(_QtLogHandler(self.log))
        return logger

    # ---------------- 主流程 ----------------
    def run(self):
        self.stateChanged.emit("starting")
        self._started_at = time.time()
        enable_dpi_awareness()
        logger = self._make_logger()
        try:
            self._run_loop(logger)
        except Exception as exc:                       # noqa: BLE001
            logger.exception(f"运行异常：{exc}")
            self.log(f"运行异常：{exc}", "error")
            self.stateChanged.emit("error")
        self.stateChanged.emit("stopped")

    def _run_loop(self, logger):
        startup_checks(logger)
        screen = Screen()
        human = Human(pyautogui.size().width, pyautogui.size().height)
        writer = ExcelWriter(CONFIG.path("output.excel_path"),
                             str(CONFIG.get("output.sheet_name", "候选达人")),
                             bool(CONFIG.get("output.excel_freeze_header", True)),
                             int(CONFIG.get("safety.dedup_memory", 400)))
        self.stats["rows"] = writer.count
        self.statsChanged.emit(dict(self.stats))

        win = self._ensure_window(logger)
        if win is None:
            screen.close()
            self.stateChanged.emit("error")
            return

        hwnd = win.hwnd
        self._switch_feed(win, screen, human)

        self.stateChanged.emit("running")
        self.log("开始运行：识别 → 过滤 → 记录（在右侧面板可随时改关键词和阈值）")

        last_fp = ""
        same_count = 0
        bad_frames = 0
        max_min = int(CONFIG.get("safety.max_runtime_minutes", 0) or 0)
        paused_announced = False

        while not self._stop:
            # ---- 暂停 ----
            if self._pause:
                if not paused_announced:
                    self.log("已暂停（画面与鼠标操作全部停止）", "warn")
                    paused_announced = True
                time.sleep(0.5)
                continue
            if paused_announced:
                paused_announced = False
                self.log("继续运行", "info")

            # ---- 热更新 ----
            changed = CONFIG.refresh()
            if changed:
                self.log(f"🔄 配置已热更新：{'、'.join(changed)}", "warn")

            # ---- 运行时长 ----
            if max_min and (time.time() - self._started_at) / 60.0 >= max_min:
                self.log(f"已达最长运行时长 {max_min} 分钟，自动结束")
                break

            # ---- 窗口健康 ----
            if not is_alive(hwnd):
                self.log("抖音窗口已关闭，工作流停止", "error")
                break
            if is_minimized(hwnd):
                self.log("抖音窗口被最小化，自动还原", "warn")
                show_no_activate(hwnd, activate=True)
                time.sleep(1.2)
                continue
            rect = refresh_rect(hwnd)
            if rect:
                win = WinInfo(hwnd, win.title, rect)

            if CONFIG.get("window.keep_foreground", True) and not activate(hwnd):
                time.sleep(1.5)
                continue

            # ---- 抽帧 ----
            try:
                f1 = grab_window(win, screen)
            except Exception as exc:
                self.log(f"截屏失败：{exc}", "warn")
                time.sleep(1.0)
                continue
            if is_bad_frame(f1):
                bad_frames += 1
                if bad_frames % 5 == 0:
                    self.log(f"画面暂时无效（第 {bad_frames} 次），等待渲染…", "warn")
                if bad_frames >= 40:
                    self.log("长时间黑屏：抖音可能没有登录或页面异常，请手动看一眼客户端", "error")
                    break
                time.sleep(1.0)
                continue
            bad_frames = 0
            frames = [f1]
            n_frames = max(1, int(CONFIG.get("vision.frames_per_video", 1) or 1))
            for _ in range(n_frames - 1):
                time.sleep(rnd(CONFIG.get("capture.sample_interval", [0.8, 1.4])))
                try:
                    fx = grab_window(win, screen)
                    if not is_bad_frame(fx):
                        frames.append(fx)
                except Exception:
                    pass

            # ---- 换视频检测 ----
            fp = frame_fingerprint(frames[-1])
            if last_fp and hamming_hex(last_fp, fp) <= 3:
                same_count += 1
                if same_count >= int(CONFIG.get("safety.same_frame_retry", 3)):
                    self.log("画面连续未变化，换滚轮方式重试", "warn")
                    human.scroll_next(win, method="wheel")
                    same_count = 0
                    time.sleep(1.0)
                    continue
            else:
                same_count = 0
            last_fp = fp

            # ---- 先把画面推给界面（预览实时感） ----
            self.frameReady.emit(frames[0])

            # ---- 识别 ----
            self.stats["idx"] += 1
            try:
                result = process_one(win, screen, frames, logger)
            except Exception as exc:
                logger.exception(f"处理异常：{exc}")
                self.log(f"本条视频处理异常，跳过：{exc}", "error")
                time.sleep(1.0)
                continue

            decision = result["decision"]
            nickname = result["nickname"]
            metrics = result["metrics"]

            self.frameReady.emit(frames[-1])
            self.log("─" * 56, "debug")
            self.log(f"[#{self.stats['idx']:>3}] {decision.grade} | {nickname or '未知达人'} | "
                     f"眼部 {result['vision'].lens_suitable:.0f} 分 | "
                     f"❤{fmt_num(metrics.like)} 💬{fmt_num(metrics.comment)} | "
                     f"综合分 {decision.score}")

            # ---- 落表 ----
            write_it = False
            if decision.is_candidate:
                write_it = True
                if decision.grade == GRADE_HIGH:
                    self.stats["high"] += 1
                else:
                    self.stats["normal"] += 1
            elif decision.grade == GRADE_SPAM:
                self.stats["spam"] += 1
                if CONFIG.get("filter.record_spam_to_excel", True):
                    write_it = True
            else:
                self.stats["reject"] += 1
                if CONFIG.get("filter.record_rejected_to_excel", False):
                    write_it = True

            if write_it:
                if writer.append(result["record"], nickname, fp[:16]):
                    self.stats["rows"] = writer.count
                if writer.last_warning():
                    self.log(writer.last_warning(), "warn")

            self.statsChanged.emit(dict(self.stats))
            self.resultReady.emit({
                "grade": decision.grade,
                "score": decision.score,
                "nickname": nickname,
                "reason": decision.reason_text,
                "hit": decision.hit_keywords,
                "tags": result["hashtags"],
                "search": result["search_confirmed"],
                "like": metrics.like, "comment": metrics.comment,
                "play": metrics.play, "like_rate": decision.like_rate,
                "link": result["record"].get("video_link", ""),
                "wrote": write_it,
            })

            # ---- 自动点赞 ----
            if (CONFIG.get("auto_like.enabled", False)
                    and decision.grade in (GRADE_HIGH, GRADE_NORMAL)
                    and not decision.is_spam and human.can_like()):
                ok, note = human.click_like(win, frames[0])
                if ok:
                    self.stats["liked"] += 1
                    self.log(f"👍 已点赞（{note}）— {nickname or '未知达人'}  本场第 {self.stats['liked']} 次")
                    time.sleep(rnd(CONFIG.get("auto_like.delay_after_like", [3, 6])))
                else:
                    self.log(f"自动点赞跳过：{note}", "warn")

            self.statsChanged.emit(dict(self.stats))

            # ---- 停顿 + 下滑 ----
            time.sleep(rnd(CONFIG.get("rhythm.dwell_seconds", [2, 5])))
            if human.maybe_rest():
                self.log("拟人休息结束，继续")
            if self._stop:
                break
            human.scroll_next(win)

        screen.close()
        if writer.count or self.stats["idx"]:
            mins = (time.time() - self._started_at) / 60.0
            self.log("═" * 56)
            self.log(f"运行结束：识别 {self.stats['idx']} 条 | 高匹配 {self.stats['high']} | "
                     f"普通 {self.stats['normal']} | 刷量 {self.stats['spam']} | "
                     f"不达标 {self.stats['reject']} | 点赞 {self.stats['liked']} 次 | "
                     f"时长 {mins:.1f} 分钟")
            self.log(f"结果表格：{CONFIG.path('output.excel_path')}")

    # ---------------- 启动子流程 ----------------
    def _ensure_window(self, logger):
        kw = CONFIG.get("window.title_keywords", [])
        proc = CONFIG.get("window.process_name", "")
        win, note = None, ""
        launched = False

        for attempt in range(2):
            win, note = resolve_douyin_window(kw, proc, True)
            if win:
                break
            if note == "not_running" and CONFIG.get("window.launch_if_not_running", True):
                exe = find_douyin_exe(CONFIG.get("window.executable", ""))
                if exe:
                    self.log(f"抖音客户端未运行，正在自动启动：{exe}")
                    launch_douyin(exe)
                    launched = True
                    deadline = time.time() + 60
                    while time.time() < deadline and not self._stop:
                        time.sleep(2.0)
                        win, note = resolve_douyin_window(kw, proc, True)
                        if win:
                            break
                        if time.time() > deadline - 20:
                            from src.winutil import open_douyin_via_tray
                            self.log("尝试从系统托盘唤出主窗口…")
                            open_douyin_via_tray()
                else:
                    self.log("未找到抖音客户端安装路径，请手动启动抖音后重试", "error")
                    return None
            if not win and not launched:
                time.sleep(1.0)

        if not win:
            self.log(f"无法定位抖音窗口（{note}）。请手动打开抖音客户端，再点「启动工作流」", "error")
            return None

        self.log(f"已锁定抖音窗口 {win.width}x{win.height}（{note}）")
        activate(win.hwnd)

        # 等画面真正渲染出来（黑屏重试最多 25 秒）
        screen = Screen()
        try:
            deadline = time.time() + 25
            while time.time() < deadline and not self._stop:
                fr = grab_window(win, screen)
                if not is_bad_frame(fr):
                    self.frameReady.emit(fr)
                    break
                time.sleep(1.0)
            else:
                self.log("窗口画面迟迟未渲染（黑屏），请确认抖音已登录", "error")
                return None
        finally:
            screen.close()
        return win

    def _switch_feed(self, win, screen, human):
        if not CONFIG.get("window.auto_switch_feed", True):
            return
        tab = CONFIG.get("window.feed_tab", "推荐")
        targets = [str(t) for t in tab] if isinstance(tab, (list, tuple)) else [str(tab)]
        activate(win.hwnd)
        time.sleep(1.0)
        try:
            ok, msg = human.click_text(win, screen, targets)
            self.log(("导航：" if ok else "自动切推荐页未成功（不影响运行）：") + msg,
                     "info" if ok else "warn")
        except Exception as exc:
            self.log(f"切换推荐页异常（不影响运行）：{exc}", "warn")


# ===========================================================================
# 主界面
# ===========================================================================
class MainWindow(QMainWindow):
    TABLE_HEADERS = ["时间", "达人昵称", "等级", "命中关键词", "点赞", "评论",
                     "播放", "点赞率", "判定理由"]

    def __init__(self):
        super().__init__()
        self.worker: Worker | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._build_ui()
        self._load_keywords_to_editor()
        self._load_settings()
        self._poll_status()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(1500)

    # ------------------------------------------------------------ UI 构建
    def _build_ui(self):
        self.setWindowTitle("抖音美瞳达人筛选 · 可视化控制台")
        self.resize(1180, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # ================= 顶栏 =================
        bar = QHBoxLayout()
        self.btn_start = QPushButton("▶  启动工作流")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_worker)

        self.btn_pause = QPushButton("⏸  暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("⏹  停止")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_worker)

        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addSpacing(14)

        self.status_dot = QLabel("●")
        self.status_text = QLabel("抖音窗口：检测中…")
        bar.addWidget(self.status_dot)
        bar.addWidget(self.status_text)
        bar.addStretch(1)

        self.btn_show = QPushButton("唤出窗口")
        self.btn_show.clicked.connect(self.show_douyin_window)
        self.btn_feed = QPushButton("切到推荐页")
        self.btn_feed.clicked.connect(self.switch_feed_clicked)
        self.btn_excel = QPushButton("打开结果表")
        self.btn_excel.clicked.connect(self.open_excel)
        self.btn_logs = QPushButton("打开日志")
        self.btn_logs.clicked.connect(self.open_logs)
        for b in (self.btn_show, self.btn_feed, self.btn_excel, self.btn_logs):
            bar.addWidget(b)
        root.addLayout(bar)

        # ================= 中部左右分栏 =================
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- 左：实时画面 + 结果卡 ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        g_preview = QGroupBox("实时画面")
        pv = QVBoxLayout(g_preview)
        self.preview = QLabel("等待启动…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(520, 300)
        self.preview.setStyleSheet(
            "background:#101215;border:2px solid #2A2F38;border-radius:8px;color:#565E6A;font-size:14px;")
        pv.addWidget(self.preview)
        lv.addWidget(g_preview)

        g_card = QGroupBox("当前识别结果")
        cv = QGridLayout(g_card)
        cv.setVerticalSpacing(6)

        self.badge = QLabel("待机")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedWidth(88)
        self.badge.setStyleSheet(
            f"background:{GRADE_COLOR[GRADE_REJECT]};color:white;border-radius:5px;"
            "font-weight:700;padding:5px 0;")
        self.nickname = QLabel("—")
        f = QFont(); f.setBold(True); f.setPointSize(13)
        self.nickname.setFont(f)
        self.metrics_lbl = QLabel("—")
        self.keywords_lbl = QLabel("—")
        self.keywords_lbl.setWordWrap(True)
        self.reason_lbl = QLabel("点击「启动工作流」后，这里会实时显示每条视频的识别结论")
        self.reason_lbl.setWordWrap(True)
        self.reason_lbl.setStyleSheet("color:#9AA3B0;")
        cv.addWidget(self.badge, 0, 0)
        cv.addWidget(self.nickname, 0, 1)
        cv.addWidget(self.metrics_lbl, 1, 1)
        cv.addWidget(self.keywords_lbl, 2, 1)
        cv.addWidget(self.reason_lbl, 3, 0, 1, 2)
        cv.setColumnStretch(1, 1)
        lv.addWidget(g_card)
        splitter.addWidget(left)

        # ---- 右：统计 + 设置 ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        g_stat = QGroupBox("运行统计")
        sg = QGridLayout(g_stat)
        sg.setVerticalSpacing(4)
        self.stat_labels: dict[str, QLabel] = {}
        items = [("idx", "已识别"), ("high", "高匹配"), ("normal", "普通"),
                 ("spam", "刷量号"), ("reject", "不达标"), ("liked", "已点赞")]
        for i, (key, name) in enumerate(items):
            r, c = divmod(i, 2)
            box = QVBoxLayout()
            val = QLabel("0")
            val.setObjectName("statValue")
            val.setAlignment(Qt.AlignCenter)
            cap = QLabel(name)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("color:#8B93A1;font-size:11px;")
            box.addWidget(val)
            box.addWidget(cap)
            sg.addLayout(box, r, c)
            self.stat_labels[key] = val
        self.elapsed_lbl = QLabel("0:00")
        self.elapsed_lbl.setObjectName("statValue")
        self.elapsed_lbl.setAlignment(Qt.AlignCenter)
        ecap = QLabel("运行时长")
        ecap.setAlignment(Qt.AlignCenter)
        ecap.setStyleSheet("color:#8B93A1;font-size:11px;")
        eg = QVBoxLayout()
        eg.addWidget(self.elapsed_lbl)
        eg.addWidget(ecap)
        sg.addLayout(eg, 3, 0, 1, 2)
        rv.addWidget(g_stat)

        g_set = QGroupBox("快捷设置（改完即生效，无需重启）")
        sv = QVBoxLayout(g_set)

        self.chk_like = QCheckBox("自动点赞（满足全部条件才点）")
        self.chk_like.toggled.connect(self._on_like_toggle)
        sv.addWidget(self.chk_like)
        warn = QLabel("⚠ 开启后有抖音风控 / 限流 / 封号风险，风险自担")
        warn.setObjectName("hint")
        sv.addWidget(warn)

        row = QHBoxLayout()
        row.addWidget(QLabel("点赞率阈值"))
        self.spin_rate = QDoubleSpinBox()
        self.spin_rate.setRange(0.0, 0.5)
        self.spin_rate.setDecimals(3)
        self.spin_rate.setSingleStep(0.005)
        self.spin_rate.setToolTip("点赞率 = 点赞 ÷ 播放，低于该值判为刷量号。\n推荐流拿不到播放量时该规则自动跳过。")
        self.spin_rate.valueChanged.connect(self._on_rate_changed)
        row.addWidget(self.spin_rate)
        row.addStretch(1)
        sv.addLayout(row)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("视觉服务商"))
        self.cmb_provider = QComboBox()
        self.cmb_provider.addItems([p[0] for p in PROVIDER_PRESETS])
        self.cmb_provider.setToolTip("切换后会自动写入对应的 base_url 和 model（config.yaml 里也能手改）")
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        row3.addWidget(self.cmb_provider, 1)
        sv.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("视觉 API Key"))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-...（DeepSeek Key）")
        row4.addWidget(self.key_edit, 1)
        self.btn_key_save = QPushButton("保存")
        self.btn_key_save.clicked.connect(self.save_api_key)
        row4.addWidget(self.btn_key_save)
        self.btn_key_test = QPushButton("测试")
        self.btn_key_test.clicked.connect(self.test_api_key)
        row4.addWidget(self.btn_key_test)
        sv.addLayout(row4)
        key_hint = QLabel("不填或 Key 无效 = 只有本地降级，无法判断颜值/美瞳适合度")
        key_hint.setObjectName("hint")
        sv.addWidget(key_hint)

        sv.addWidget(QLabel("关键词库（一行一个，# 开头为注释，留空 = 不限关键词）"))
        self.kw_edit = QPlainTextEdit()
        self.kw_edit.setFixedHeight(130)
        sv.addWidget(self.kw_edit)
        row2 = QHBoxLayout()
        self.kw_count = QLabel("")
        self.kw_count.setObjectName("hint")
        self.btn_kw_save = QPushButton("保存关键词库")
        self.btn_kw_save.clicked.connect(self.save_keywords)
        row2.addWidget(self.kw_count)
        row2.addStretch(1)
        row2.addWidget(self.btn_kw_save)
        sv.addLayout(row2)

        self.chk_theme = QCheckBox("美瞳内容自动识别（AI 挑出在卖/种美瞳的视频）")
        self.chk_theme.toggled.connect(self._on_theme_toggle)
        sv.addWidget(self.chk_theme)
        theme_hint = QLabel("开启后：关键词库为空时，只有被判定为美瞳相关"
                            "（画面文字 / 话题标签 / 文案里有美瞳字样）的视频才会记录，其余跳过")
        theme_hint.setObjectName("hint")
        theme_hint.setWordWrap(True)
        sv.addWidget(theme_hint)
        rv.addWidget(g_set)
        rv.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ================= 底部页签 =================
        tabs = QTabWidget()
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(3000)
        tabs.addTab(self.log_panel, "运行日志")

        self.table = QTableWidget(0, len(self.TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._open_row_link)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(len(self.TABLE_HEADERS) - 1, QHeaderView.Stretch)
        tabs.addTab(self.table, "候选达人（双击打开链接）")
        tabs.setCurrentIndex(0)
        root.addWidget(tabs, 1)

        self._log(f"工作目录：{ROOT_DIR}")
        self._log("提示：点「启动工作流」即可——会自动打开抖音、切到推荐页并开始识别。")

    # ------------------------------------------------------------ 关键词/设置
    def _load_keywords_to_editor(self):
        path = CONFIG.path("keywords.file", "keywords.txt")
        text = ""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    text = f.read()
            except OSError:
                text = ""
        self.kw_edit.setPlainText(text)
        self._update_kw_count()

    def _update_kw_count(self):
        words = [ln.strip() for ln in self.kw_edit.toPlainText().splitlines()
                 if ln.strip() and not ln.strip().startswith(("#", "//"))]
        self.kw_count.setText(f"当前 {len(words)} 个词" + ("（空 = 不限关键词）" if not words else ""))

    def save_keywords(self):
        path = CONFIG.path("keywords.file", "keywords.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.kw_edit.toPlainText().rstrip() + "\n")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        CONFIG.refresh()
        self._update_kw_count()
        lib = CONFIG.keywords
        self._log(f"关键词库已保存：{len(lib.words)} 个词（下一轮识别生效）",
                  "info" if not lib.empty else "warn")

    def _load_settings(self):
        self.spin_rate.blockSignals(True)
        self.spin_rate.setValue(float(CONFIG.get("filter.like_rate_threshold", 0.02)))
        self.spin_rate.blockSignals(False)
        self.chk_like.blockSignals(True)
        self.chk_like.setChecked(bool(CONFIG.get("auto_like.enabled", False)))
        self.chk_like.blockSignals(False)
        self.chk_theme.blockSignals(True)
        self.chk_theme.setChecked(bool(CONFIG.get("theme_detect.enabled", True)))
        self.chk_theme.blockSignals(False)
        self.key_edit.setText(str(CONFIG.get("vision.api_key", "") or ""))

        # 服务商下拉：按当前 base_url 反推该选哪一项
        base = str(CONFIG.get("vision.base_url", "") or "").lower()
        prov = str(CONFIG.get("vision.provider", "") or "").lower()
        idx = 2
        if "dashscope" in base or "aliyun" in base or prov == "dashscope":
            idx = 1
        elif "deepseek" in base or prov == "deepseek":
            idx = 0
        self.cmb_provider.blockSignals(True)
        self.cmb_provider.setCurrentIndex(idx)
        self.cmb_provider.blockSignals(False)
        self.key_edit.setPlaceholderText(PROVIDER_PRESETS[idx][4])

    def _on_provider_changed(self, idx: int):
        if idx < 0 or idx >= len(PROVIDER_PRESETS):
            return
        name, pid, base, model, placeholder = PROVIDER_PRESETS[idx]
        self.key_edit.setPlaceholderText(placeholder)
        if pid == "custom":
            self._log("已切换到自定义服务商：base_url / model 请自行到 config.yaml 里改", "warn")
            return
        _update_yaml_value("vision.provider", f'"{pid}"')
        _update_yaml_value("vision.base_url", f'"{base}"')
        _update_yaml_value("vision.model", f'"{model}"')
        self._log(f"视觉服务商已切换为 {pid}：{model} @ {base}（建议点「测试」）")

    def _on_theme_toggle(self, on: bool):
        if _update_yaml_value("theme_detect.enabled", bool(on)):
            self._log("美瞳内容自动识别已" + ("开启：只记录判定为美瞳相关的视频" if on
                                            else "关闭：恢复为仅按关键词库判断"))

    def save_api_key(self):
        key = self.key_edit.text().strip()
        if _update_yaml_value("vision.api_key", f'"{key}"'):
            self._log("API Key 已保存" + ("（建议点「测试」验证有效性）" if key else "（已清空，将使用本地降级）"),
                      "info" if key else "warn")

    def test_api_key(self):
        # 先把输入框里的值存进去再测，避免"填了没保存就测试"的困惑
        self.save_api_key()
        self.btn_key_test.setEnabled(False)
        self.btn_key_test.setText("测试中…")
        self._tester = ApiTester()
        self._tester.done.connect(self._on_api_test)
        self._tester.start()

    def _on_api_test(self, ok: bool, msg: str):
        self.btn_key_test.setEnabled(True)
        self.btn_key_test.setText("测试")
        self._log(("✅ 视觉API：" if ok else "❌ 视觉API：") + msg, "info" if ok else "error")
        if not ok:
            QMessageBox.warning(self, "API 连接失败", msg +
                                "\n\n排查建议：\n"
                                "1) 确认「视觉服务商」和下面对话框里的 Key 是同一家的\n"
                                "   （DeepSeek 的 Key 配阿里云的地址会直接 401）\n"
                                "2) DeepSeek：到 platform.deepseek.com 看 Key 是否有效、账户是否还有余额\n"
                                "3) 模型名要写对 —— DeepSeek 的视觉模型是 deepseek-flash\n"
                                "   （deepseek-v4-pro 不支持图片输入）")

    def _on_rate_changed(self, val: float):
        if _update_yaml_value("filter.like_rate_threshold", round(float(val), 4)):
            self._log(f"点赞率阈值已改为 {val:.2%}（下一轮识别生效）")

    def _on_like_toggle(self, on: bool):
        if _update_yaml_value("auto_like.enabled", bool(on)):
            if on:
                self._log("自动点赞已开启：满足全部条件后才会点击，且有风控风险", "warn")
            else:
                self._log("自动点赞已关闭（只识别，不点任何按钮）")

    # ------------------------------------------------------------ 状态灯
    def _poll_status(self):
        st = window_status(CONFIG.get("window.process_name", "douyin.exe"))
        mapping = {
            "ok": ("#30A46C", "抖音窗口：正常"),
            "minimized": ("#F5A623", "抖音窗口：最小化（启动后自动还原）"),
            "hidden": ("#F5A623", "抖音窗口：在托盘（启动后自动调出）"),
            "not_running": ("#E5484D", "抖音客户端：未运行（启动后自动拉起）"),
        }
        color, text = mapping.get(st, ("#E5484D", "抖音窗口：未知"))
        self.status_dot.setStyleSheet(f"color:{color};font-size:15px;")
        self.status_text.setText(text)
        self.status_text.setStyleSheet(f"color:{color};" if st != "ok" else "color:#30A46C;")

    # ------------------------------------------------------------ 启停控制
    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = Worker()
        self.worker.logLine.connect(self._on_worker_log)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.resultReady.connect(self._on_result)
        self.worker.statsChanged.connect(self._on_stats)
        self.worker.stateChanged.connect(self._on_state)
        self.worker.start()
        self._elapsed_timer.start(1000)

    def toggle_pause(self):
        if not self.worker:
            return
        new_state = not self.worker._pause
        self.worker.set_pause(new_state)
        self.btn_pause.setText("▶  继续" if new_state else "⏸  暂停")

    def stop_worker(self):
        if not self.worker:
            return
        self.worker.request_stop()
        self._log("正在停止…（等当前动作收尾，最多 20 秒）", "warn")
        # 兜底：若 20 秒内没自然退出（如网络请求卡住），强制结束线程
        QTimer.singleShot(20000, self._force_terminate)

    def _force_terminate(self):
        w = self.worker
        if w and w.isRunning():
            w.terminate()
            w.wait(3000)
            self._log("已强制停止", "warn")
            self._on_state("stopped")

    def _on_state(self, state: str):
        running = state in ("starting", "running", "paused")
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        if state == "stopped" or state == "error":
            self.btn_pause.setText("⏸  暂停")
            self._elapsed_timer.stop()

    # ------------------------------------------------------------ 回调
    def _on_worker_log(self, text: str, level: str):
        self._log(text, level)

    def _on_frame(self, arr: np.ndarray):
        try:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(img).scaled(
                self.preview.width() - 4, self.preview.height() - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(pm)
        except Exception:
            pass

    def _on_result(self, r: dict):
        grade = r["grade"]
        color = GRADE_COLOR.get(grade, "#6E7681")
        self.badge.setText(grade)
        self.badge.setStyleSheet(
            f"background:{color};color:white;border-radius:5px;font-weight:700;padding:5px 0;")
        self.preview.setStyleSheet(
            f"background:#101215;border:2px solid {color};border-radius:8px;")
        self.nickname.setText(r["nickname"] or "未知达人")
        rate = f"{r['like_rate']:.2%}" if r.get("like_rate") is not None else "N/A"
        self.metrics_lbl.setText(
            f"❤ {fmt_num(r['like'])}    💬 {fmt_num(r['comment'])}    "
            f"▶ {fmt_num(r['play'])}    点赞率 {rate}    综合分 {r['score']}")
        kws = "、".join(r["hit"]) if r.get("hit") else "—"
        tags = " ".join("#" + t for t in (r.get("tags") or [])[:6])
        search = " / ".join((r.get("search") or [])[:3])
        self.keywords_lbl.setText(f"🎯 命中：{kws}" + (f"    🔍 {search}" if search else "")
                                  + (f"\n🏷 {tags}" if tags else ""))
        self.reason_lbl.setText(r["reason"] or "—")

        if r.get("wrote"):
            self._insert_row(r)

    def _insert_row(self, r: dict):
        if self.table.rowCount() >= 500:
            self.table.removeRow(self.table.rowCount() - 1)
        self.table.insertRow(0)
        color = QColor(GRADE_COLOR.get(r["grade"], "#6E7681"))
        rate = f"{r['like_rate']:.2%}" if r.get("like_rate") is not None else ""
        values = [datetime.now().strftime("%H:%M:%S"), r["nickname"] or "未知达人", r["grade"],
                  "、".join(r["hit"]), fmt_num(r["like"]) if r["like"] is not None else "",
                  fmt_num(r["comment"]) if r["comment"] is not None else "",
                  fmt_num(r["play"]) if r["play"] is not None else "",
                  rate, r["reason"] or ""]
        for col, v in enumerate(values):
            it = QTableWidgetItem(str(v))
            if col == 2:
                it.setForeground(color)
            if col in (4, 5, 6, 7):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(0, col, it)
        self.table.item(0, 0).setData(Qt.UserRole, r.get("link", ""))

    def _open_row_link(self, index):
        it = self.table.item(index.row(), 0)
        link = it.data(Qt.UserRole) if it else ""
        if link:
            QDesktopServices.openUrl(QUrl(link))

    def _on_stats(self, s: dict):
        for key, lbl in self.stat_labels.items():
            lbl.setText(str(s.get(key, 0)))

    def _tick_elapsed(self):
        if self.worker and self.worker._started_at:
            secs = int(time.time() - self.worker._started_at)
            self.elapsed_lbl.setText(f"{secs // 60}:{secs % 60:02d}")

    # ------------------------------------------------------------ 按钮
    def show_douyin_window(self):
        win, note = resolve_douyin_window(CONFIG.get("window.title_keywords", []),
                                          CONFIG.get("window.process_name", ""), True)
        if win:
            activate(win.hwnd)
            self._log(f"已唤出抖音窗口（{note}）")
        else:
            exe = find_douyin_exe(CONFIG.get("window.executable", ""))
            if exe:
                launch_douyin(exe)
                self._log("抖音客户端未运行，已自动启动，几秒后窗口会出现")
            else:
                self._log("没找到抖音客户端，请先安装 / 启动", "error")
        self._poll_status()

    def switch_feed_clicked(self):
        if self.worker and self.worker.isRunning():
            self._log("工作流运行中会自动保持在推荐页，无需手动切换", "warn")
            return
        win, note = resolve_douyin_window(CONFIG.get("window.title_keywords", []),
                                          CONFIG.get("window.process_name", ""), True)
        if not win:
            self._log(f"未找到抖音窗口（{note}）", "error")
            return
        activate(win.hwnd)
        screen = Screen()
        try:
            human = Human(pyautogui.size().width, pyautogui.size().height)
            ok, msg = human.click_text(win, screen, [str(CONFIG.get("window.feed_tab", "推荐"))])
            self._log(("已" if ok else "未") + "切换到推荐页：" + msg, "info" if ok else "warn")
        finally:
            screen.close()

    def open_excel(self):
        p = CONFIG.path("output.excel_path")
        if os.path.isfile(p):
            os.startfile(p)
        else:
            self._log("结果表还没有生成，跑出第一条候选达人后自动创建", "warn")

    def open_logs(self):
        os.startfile(CONFIG.path("output.log_dir", "logs"))

    # ------------------------------------------------------------ 日志
    def _log(self, text: str, level: str = "info"):
        icon = {"info": "·", "warn": "⚠", "error": "✖", "debug": " "}.get(level, "·")
        self.log_panel.appendPlainText(f"{datetime.now():%H:%M:%S} {icon} {text}")

    # ------------------------------------------------------------ 退出
    def closeEvent(self, ev):
        if self.worker and self.worker.isRunning():
            ret = QMessageBox.question(self, "退出确认",
                                       "工作流正在运行，确定停止并退出吗？",
                                       QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(8000)


def main():
    enable_dpi_awareness()
    pyautogui.FAILSAFE = True
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        err = traceback.format_exc()
        try:
            with open(os.path.join(BASE_DIR, "logs", "gui_error.log"), "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now()}]\n{err}\n")
        except OSError:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, err[-1500:], "界面启动失败", 0x10)
        except Exception:
            pass
        raise
