# -*- coding: utf-8 -*-
"""Windows 窗口定位、前景控制、画面捕获。

全部走 ctypes，不依赖 pywin32，减少安装体积与兼容性问题。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mss
import numpy as np

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ------------------------------------------------------------------ DPI 感知
def enable_dpi_awareness() -> None:
    """让 GetWindowRect / mss 使用同一套物理像素坐标，避免 125%/150% 缩放下错位。"""
    try:
        # Windows 10 1703+: PER_MONITOR_AWARE_V2
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


# ------------------------------------------------------------------ 结构体
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class WINDOWPLACEMENT(ctypes.Structure):
    """最小化窗口的 GetWindowRect 会返回任务栏按钮的尺寸（形如 160x28），
    必须用 GetWindowPlacement 的 rcNormalPosition 才能拿到「还原后」的位置。"""
    _fields_ = [("length", ctypes.c_uint), ("flags", ctypes.c_uint),
                ("showCmd", ctypes.c_uint), ("ptMinPosition", POINT),
                ("ptMaxPosition", POINT), ("rcNormalPosition", RECT)]


SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_MINIMIZE = 6
SW_SHOWNA = 8
SW_RESTORE = 9


@dataclass
class WinInfo:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]   # left, top, right, bottom（物理像素）
    visible: bool = True              # False = 窗口被隐藏（缩在托盘里）
    minimized: bool = False           # True = 窗口被最小化到任务栏

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    def crop_box(self, region: List[float]) -> Tuple[int, int, int, int]:
        """把 [0~1] 相对区域换算成窗口内的绝对像素框。"""
        l, t, r, b = region
        x0 = self.rect[0] + int(self.width * l)
        y0 = self.rect[1] + int(self.height * t)
        x1 = self.rect[0] + int(self.width * r)
        y1 = self.rect[1] + int(self.height * b)
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


# ------------------------------------------------------------------ 进程名
def _pid_to_exe(pid: int) -> str:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value).lower()
    finally:
        kernel32.CloseHandle(h)
    return ""


# ------------------------------------------------------------------ 枚举
def list_windows(include_untitled: bool = False, include_hidden: bool = False) -> List[WinInfo]:
    out: List[WinInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def _cb(hwnd, _lparam):
        visible = bool(user32.IsWindowVisible(hwnd))
        if not visible and not include_hidden:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title = ""
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
        if not title and not include_untitled:
            return True
        if title in ("Default IME", "MSCTFIME UI"):
            return True
        r = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        wpx, hpx = r.right - r.left, r.bottom - r.top

        # 【关键】最小化窗口的 GetWindowRect 只有任务栏按钮那么大（如 160x28），
        # 用尺寸过滤会把真正的主窗口误判为"不存在"。这里改用还原后的正常位置。
        iconic = bool(user32.IsIconic(hwnd))
        if iconic:
            pl = WINDOWPLACEMENT()
            pl.length = ctypes.sizeof(WINDOWPLACEMENT)
            if user32.GetWindowPlacement(hwnd, ctypes.byref(pl)):
                nr = pl.rcNormalPosition
                if nr.right - nr.left > 0 and nr.bottom - nr.top > 0:
                    r = RECT(nr.left, nr.top, nr.right, nr.bottom)
                    wpx, hpx = nr.right - nr.left, nr.bottom - nr.top

        if wpx < 200 or hpx < 200:
            return True
        out.append(WinInfo(int(hwnd), title, (r.left, r.top, r.right, r.bottom), visible, iconic))
        return True

    user32.EnumWindows(_cb, 0)
    return out


def show_no_activate(hwnd: int, restore: bool = True, activate: bool = False) -> None:
    """把隐藏/最小化的窗口显示出来，默认【不抢焦点】（SW_SHOWNOACTIVATE）。

    这样工作流可以在不打断你当前操作的前提下把抖音窗口调回桌面。
    """
    SW_HIDE = 0
    SW_SHOWNOACTIVATE = 4
    SW_SHOWNA = 8
    SW_RESTORE = 9
    if activate:
        user32.ShowWindow(hwnd, SW_RESTORE if is_minimized(hwnd) else SW_SHOWNA)
    else:
        user32.ShowWindow(hwnd, SW_RESTORE if is_minimized(hwnd) else SW_SHOWNOACTIVATE)
        if restore:
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)  # NOSIZE|NOMOVE|SHOWWINDOW


def hide_window(hwnd: int) -> None:
    """把窗口重新藏起来（把你原来的托盘状态还原回去）。"""
    user32.ShowWindow(hwnd, 0)


def window_exe(hwnd: int) -> str:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return _pid_to_exe(pid.value)


def find_douyin_window(title_keywords: List[str], process_name: str = "",
                       allow_hidden: bool = False) -> Optional[WinInfo]:
    """定位抖音客户端窗口。

    判定权重：进程名命中(5) > 标题含有客户端特征词(3) > 标题命中关键词(2)。
    抖音 PC 客户端是 Electron 应用，主窗口标题通常就是 "douyin"（或当前页面名），
    所以【进程名才是主信号】，标题只作辅助。

    allow_hidden=True 时，连被隐藏（缩在托盘）的窗口也一并返回。
    """
    kw = [k.strip().lower() for k in (title_keywords or []) if k and k.strip()]
    proc = (process_name or "").strip().lower()
    hits: List[Tuple[int, WinInfo]] = []

    # 客户端内部页面窗口：不应被当成主播放界面
    inner_pages = {"设置", "活动广场", "我的", "消息", "搜索", "登录", "图片和视频", "播放器"}

    for w in list_windows(include_untitled=True, include_hidden=allow_hidden):
        exe = window_exe(w.hwnd) if proc else ""
        score = 0
        if proc and exe == proc:
            score += 5
        t = w.title.lower()
        if kw and any(k in t for k in kw):
            score += 2
        if score <= 0:
            continue
        if w.title.strip() in inner_pages:
            score -= 1
        hits.append((score, w))

    if not hits:
        return None

    # 排序偏好：进程名命中 > 可见（任务栏里也算可见）> 未最小化 > 标题非空 > 面积大
    hits.sort(key=lambda it: (it[0],
                              1 if it[1].visible else 0,
                              0 if it[1].minimized else 1,
                              1 if it[1].title else 0,
                              it[1].width * it[1].height),
              reverse=True)
    return hits[0][1]


def resolve_douyin_window(title_keywords: List[str], process_name: str = "",
                          auto_restore_hidden: bool = True) -> Tuple[Optional[WinInfo], str]:
    """一站式定位抖音主窗口，必要时自动把它调回桌面。

    覆盖三种真实情况：
      ok                     —— 窗口已经在桌面上
      restored_from_minimized—— 窗口被最小化到任务栏（最常见），已还原
      restored_from_tray     —— 窗口被隐藏（缩在托盘），已调出

    返回 (窗口信息, 说明文字)。定位失败时窗口为 None，说明文字为失败原因。
    """
    # 第一轮：只看「可见」窗口（最小化窗口 IsWindowVisible 仍为 True，所以能被找到）
    win = find_douyin_window(title_keywords, process_name, allow_hidden=False)

    # 第二轮：连被隐藏的一起看
    if win is None:
        win = find_douyin_window(title_keywords, process_name, allow_hidden=True)
    if win is None:
        return None, "not_running"

    if win.visible and not win.minimized:
        return win, "ok"

    if not auto_restore_hidden:
        return None, ("minimized" if win.minimized else "hidden_in_tray")

    if win.minimized:
        # 最小化 → SW_RESTORE 还原到原来的位置和大小
        user32.ShowWindow(win.hwnd, SW_RESTORE)
        time.sleep(1.2)
        note = "restored_from_minimized"
    else:
        # 被隐藏（托盘）→ 先显示但不抢焦点
        show_no_activate(win.hwnd, activate=False)
        time.sleep(1.6)
        note = "restored_from_tray"

    rect = refresh_rect(win.hwnd)
    if (rect
            and (rect[2] - rect[0]) > 200 and (rect[3] - rect[1]) > 200
            and not is_minimized(win.hwnd)):
        return WinInfo(win.hwnd, win.title, rect, True, False), note
    return None, "restore_failed"


# ------------------------------------------------------------------ 自动启动客户端
def _lnk_target(lnk_path: str) -> str:
    """从 .lnk 快捷方式里粗解析出目标 exe 路径（不依赖 COM）。"""
    try:
        import re
        data = open(lnk_path, "rb").read()
        txt = data.decode("utf-16-le", errors="ignore")
        for m in re.findall(r"[A-Za-z]:\\[ -~]{2,180}?[Dd]ouyin\.exe", txt):
            if os.path.isfile(m):
                return m
    except Exception:
        pass
    return ""


def find_douyin_exe(configured: str = "") -> str:
    """自动查找抖音客户端 douyin.exe。返回空串表示没找到。"""
    p = (configured or "").strip().strip('"')
    if p and os.path.isfile(p):
        return p
    candidates: List[str] = []
    # 桌面快捷方式（用户桌面 + 公共桌面）
    for desk in (os.path.join(os.path.expanduser("~"), "Desktop"),
                 r"C:\Users\Public\Desktop"):
        lnk = os.path.join(desk, "抖音.lnk")
        if os.path.isfile(lnk):
            t = _lnk_target(lnk)
            if t:
                candidates.append(t)
    candidates += [
        r"C:\Program Files (x86)\ByteDance\douyin\douyin.exe",
        r"C:\Program Files\ByteDance\douyin\douyin.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Douyin", "douyin.exe"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return ""


def launch_douyin(exe: str) -> bool:
    """以独立进程启动抖音客户端（不随脚本退出而关闭）。"""
    import subprocess
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([exe], cwd=os.path.dirname(exe) or None,
                         creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ 托盘唤起
# 该客户端启动后常驻系统托盘，主窗口要靠点托盘图标唤出。
_UIA_UI = None
_UIA_ROOT = None


def _uia():
    """惰性初始化 UIAutomation（失败返回 (None, None)）。"""
    global _UIA_UI, _UIA_ROOT
    if _UIA_UI is not None:
        return _UIA_UI, _UIA_ROOT
    try:
        from comtypes.client import GetModule, CreateObject
        GetModule("UIAutomationCore.dll")
        from comtypes.gen.UIAutomationClient import CUIAutomation, IUIAutomation
        ui = CreateObject(CUIAutomation, interface=IUIAutomation)
        root = (getattr(ui, "GetRootElement", None) or getattr(ui, "GetRootElementElement"))()
        _UIA_UI, _UIA_ROOT = ui, root
    except Exception:
        _UIA_UI, _UIA_ROOT = None, None
    return _UIA_UI, _UIA_ROOT


def _uia_find_by_name(root_el, keywords, limit=8):
    """在元素子树里找 Name 命中的元素。"""
    ui, _ = _uia()
    if ui is None:
        return []
    out = []

    def walk(el, depth):
        if len(out) >= limit or depth > 16:
            return
        try:
            name = el.CurrentName or ""
        except Exception:
            return
        if name and any(k in name for k in keywords):
            out.append((name, el))
        try:
            walker = ui.CreateTreeWalker(ui.RawViewCondition)
            child = walker.GetFirstChildElement(el)
            while child and len(out) < limit:
                walk(child, depth + 1)
                child = walker.GetNextSiblingElement(child)
        except Exception:
            pass

    walk(root_el, 0)
    return out


def _taskbar_roots():
    """任务栏 + 通知区溢出弹层的根元素（用树遍历，FindAll 对部分条件不可靠）。"""
    ui, root = _uia()
    if ui is None:
        return []
    wanted = {"Shell_TrayWnd", "TopLevelWindowForOverflowXamlIsland",
              "XamlExplorerHostIslandWindow"}
    roots = []
    try:
        walker = ui.CreateTreeWalker(ui.RawViewCondition)
        child = walker.GetFirstChildElement(root)
        seen = 0
        while child and seen < 60:
            seen += 1
            try:
                cls = child.CurrentClassName or ""
                if cls in wanted:
                    roots.append(child)
            except Exception:
                pass
            child = walker.GetNextSiblingElement(child)
    except Exception:
        pass
    return roots


def open_douyin_via_tray(max_tries: int = 2) -> str:
    """通过系统托盘 / 任务栏唤出抖音主窗口（物理点击，拟人节奏）。

    返回说明文字（供日志）。
    """
    import random
    import pyautogui

    pyautogui.FAILSAFE = False
    keywords = ["抖音", "douyin", "Douyin"]

    for attempt in range(1, max_tries + 1):
        # 1) 任务栏可见区 / 已打开的溢出弹层里找抖音按钮
        for base in _taskbar_roots():
            for name, el in _uia_find_by_name(base, keywords, limit=20):
                try:
                    r = el.CurrentBoundingRectangle
                except Exception:
                    continue
                if r.right - r.left <= 0 or r.bottom - r.top <= 0:
                    continue
                cx, cy = (r.left + r.right) // 2, (r.top + r.bottom) // 2
                pyautogui.moveTo(cx, cy, duration=random.uniform(0.25, 0.5))
                time.sleep(random.uniform(0.15, 0.4))
                pyautogui.click(clicks=2, interval=0.09)   # 双击 = 打开主界面
                time.sleep(2.0)
                return f"已双击任务栏「{name}」图标 ({cx},{cy})"

        # 2) 没找到 → 展开任务栏溢出区（^ 小箭头）再找一遍
        chevron = None
        for base in _taskbar_roots():
            hits = _uia_find_by_name(base, ["显示隐藏的图标", "Show Hidden Icons", "通知", " chevron"], limit=6)
            if hits:
                chevron = hits[0][1]
                break
        if chevron is not None:
            try:
                r = chevron.CurrentBoundingRectangle
                cx, cy = (r.left + r.right) // 2, (r.top + r.bottom) // 2
                pyautogui.moveTo(cx, cy, duration=random.uniform(0.2, 0.4))
                time.sleep(random.uniform(0.1, 0.3))
                pyautogui.click()
                time.sleep(1.4)   # 等弹层展开
            except Exception:
                pass
        time.sleep(0.6)

    return "任务栏 / 托盘中未找到抖音图标"


def window_status(process_name: str = "douyin.exe") -> str:
    """快速判断抖音窗口当前状态，供界面状态灯使用。

    返回：ok / minimized / hidden / not_running
    """
    proc = (process_name or "").lower()
    if not proc:
        return "not_running"
    any_win = False
    for w in list_windows(include_untitled=True, include_hidden=True):
        if window_exe(w.hwnd) != proc:
            continue
        any_win = True
        if w.visible and not w.minimized:
            return "ok"
    if not any_win:
        return "not_running"
    for w in list_windows(include_untitled=True, include_hidden=True):
        if window_exe(w.hwnd) == proc and w.minimized:
            return "minimized"
    return "hidden"


def diagnose_windows(process_name: str = "douyin.exe") -> List[str]:
    """诊断辅助：列出与抖音相关的所有窗口，方便排查找不到窗口的原因。"""
    proc = (process_name or "").lower()
    lines: List[str] = []
    minimized: List[str] = []
    hidden: List[str] = []
    normal: List[str] = []

    for w in list_windows(include_untitled=True, include_hidden=True):
        if proc and window_exe(w.hwnd) != proc:
            continue
        desc = f"hwnd={w.hwnd} {w.width}x{w.height} 标题={w.title!r}"
        if w.visible and not w.minimized:
            normal.append(desc + "（已在桌面）")
        elif w.minimized:
            minimized.append(desc + "（已最小化到任务栏）")
        else:
            hidden.append(desc + "（被隐藏）")

    lines += normal
    lines += minimized
    lines += hidden
    if not lines:
        lines.append("没有检测到任何抖音客户端窗口 —— 客户端可能没有启动")
    elif not normal:
        if minimized:
            lines.insert(0, f"检测到 {len(minimized)} 个抖音窗口处于【最小化】状态（在任务栏上）")
        elif hidden:
            lines.insert(0, f"检测到 {len(hidden)} 个抖音窗口处于【隐藏】状态（缩在托盘里）")
    return lines


# ------------------------------------------------------------------ 状态
def is_minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def refresh_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    return (r.left, r.top, r.right, r.bottom)


def activate(hwnd: int, tries: int = 3) -> bool:
    """把窗口拉到前台。Windows 限制后台进程抢焦点，这里用 AttachThreadInput 绕过。"""
    if is_minimized(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        time.sleep(0.35)
    for _ in range(max(1, tries)):
        if is_foreground(hwnd):
            return True
        fg = user32.GetForegroundWindow()
        cur_tid = kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg, None)
        tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)
        attached = []
        try:
            for tid in (fg_tid, tgt_tid):
                if tid and tid != cur_tid and user32.AttachThreadInput(cur_tid, tid, True):
                    attached.append(tid)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            for tid in attached:
                user32.AttachThreadInput(cur_tid, tid, False)
        time.sleep(0.3)
        if is_foreground(hwnd):
            return True
    return is_foreground(hwnd)


def is_alive(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


# ------------------------------------------------------------------ 截屏
class Screen:
    """mss 封装：跨调用复用实例（每次新建开销大）。"""

    def __init__(self):
        self._sct = mss.mss()

    def grab(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """截取 (left, top, right, bottom)，返回 BGR ndarray。"""
        l, t, r, b = box
        w, h = max(1, r - l), max(1, b - t)
        shot = self._sct.grab({"left": int(l), "top": int(t), "width": int(w), "height": int(h)})
        arr = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
        return arr[:, :, ::-1].copy()  # RGB -> BGR

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def grab_window(win: WinInfo, screen: Screen) -> np.ndarray:
    return screen.grab(win.rect)


# ------------------------------------------------------------------ 弹窗
MB_OK = 0x0
MB_ICONWARNING = 0x30
MB_ICONINFO = 0x40
MB_SYSTEMMODAL = 0x1000


def alert(title: str, message: str, warning: bool = True) -> None:
    """阻塞式弹窗，用于「窗口丢失 / 需要人工介入」的暂停提醒。"""
    flags = MB_OK | MB_SYSTEMMODAL | (MB_ICONWARNING if warning else MB_ICONINFO)
    try:
        user32.MessageBoxW(None, message, title, flags)
    except Exception:
        print(f"[弹窗失败] {title}: {message}")
