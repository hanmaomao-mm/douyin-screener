# -*- coding: utf-8 -*-
"""抖音美瞳达人筛选工作流 —— 主程序。

只做三件事：读画面、做判断、记数据。
不做：自动评论、自动私信、自动关注、自动分享。
"""
from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime
from urllib.parse import quote

# 允许 `python main.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pyautogui

from src import ocr as ocr_mod
from src import vision as vision_mod
from src.config import CONFIG, ROOT_DIR
from src.excel_out import ExcelWriter
from src.human import Human, rnd
from src.judge import GRADE_HIGH, GRADE_NORMAL, GRADE_REJECT, GRADE_SPAM, judge
from src.logsetup import banner, setup_logger, write_jsonl
from src.parser import (extract_comments, extract_hashtags, extract_metrics,
                        extract_nickname, extract_search_terms)
from src.winutil import (Screen, activate, alert, diagnose_windows,
                         enable_dpi_awareness, find_douyin_exe, grab_window,
                         hide_window, is_alive, is_minimized, launch_douyin,
                         open_douyin_via_tray, refresh_rect,
                         resolve_douyin_window, show_no_activate, WinInfo)

_STOP = False
_OPTS = {"check": False, "once": False, "no_alert": False}


def _popup(title: str, message: str, warning: bool = True) -> None:
    if _OPTS["no_alert"]:
        print(f"\n[需要人工处理] {title}\n{message}\n")
        return
    alert(title, message, warning)


def _on_sigint(_sig, _frame):
    global _STOP
    _STOP = True
    print("\n[收到中断信号] 正在安全退出...")


def frame_fingerprint(img: np.ndarray, size: int = 16) -> str:
    """感知哈希：用于判断「画面有没有变化」（即滑动是否生效）。"""
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 999


def is_bad_frame(img: np.ndarray) -> bool:
    """黑屏 / 过渡帧 / 纯色帧 过滤。"""
    if img is None or img.size == 0:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < 12 or float(gray.std()) < 6


def build_links(nickname: str) -> tuple[str, str]:
    """桌面客户端拿不到真实 URL，这里生成可点击的抖音搜索链接便于人工定位。"""
    mode = str(CONFIG.get("output.link_mode", "search")).lower()
    if mode == "empty" or not nickname:
        return "", ""
    url = "https://www.douyin.com/search/" + quote(nickname)
    return url, url


def _j(v) -> str:
    if v is None:
        return ""
    return str(v)


def fmt_num(v) -> str:
    if v is None:
        return "?"
    if v >= 1e8:
        return f"{v / 1e8:.1f}亿"
    if v >= 1e4:
        return f"{v / 1e4:.1f}万"
    return str(v)


def console_line(idx: int, nickname: str, vres, decision, metrics, hashtags, search_confirmed,
                 hit_keywords) -> str:
    icon = {"高匹配": "🟥", "普通": "🟨", "刷量号": "🟩", "不达标": "⬜"}.get(decision.grade, "⬜")
    name = nickname or "未知达人"
    tags = " ".join("#" + t for t in hashtags[:4]) or "—"
    st = search_confirmed[0] if search_confirmed else "—"
    kw = "、".join(hit_keywords) if hit_keywords else "—"
    rate = f"{decision.like_rate:.2%}" if decision.like_rate is not None else "N/A"
    head = (f"[#{idx:>3}] {icon} {decision.grade:<4} | {name[:14]:<14} | "
            f"眼部{vres.lens_suitable:.0f}分 | ❤{fmt_num(metrics.like)}/💬{fmt_num(metrics.comment)} | "
            f"点赞率{rate}")
    detail = (f"          🏷 {tags}  🔍 {st}  🎯 {kw}  分{decision.score}")
    if decision.grade == GRADE_REJECT:
        detail = f"          ⛔ {decision.reason_text}"
    return head + "\n" + detail


def startup_checks(logger) -> None:
    print(banner("抖音美瞳达人筛选工作流 · 启动自检"))
    logger.info(f"项目目录：{ROOT_DIR}")

    # 1. OCR
    eng = ocr_mod.engine_status()
    logger.info(f"文字识别引擎：{eng}")
    if eng == "vision":
        logger.warning("本地 OCR 不可用，将改用多模态模型读字（更慢、更贵）")

    # 2. 视觉
    key = (CONFIG.get("vision.api_key", "") or "").strip()
    if key:
        logger.info(f"视觉模型：{CONFIG.get('vision.model')} @ {CONFIG.get('vision.base_url')}")
    else:
        logger.warning("⚠ 未配置 vision.api_key，将使用【本地启发式】判断（无法评估颜值，只能判断眼睛是否看得清）")
        logger.warning("  如需 AI 视觉识别：到 platform.deepseek.com 申请 Key，填进 vision.api_key（或用界面上的输入框）")

    # 3. 关键词库
    kw = CONFIG.keywords
    if kw.empty:
        logger.info("关键词库：空 → 不限制关键词，仅校验眼部条件并过滤刷量号")
    else:
        logger.info(f"关键词库：{len(kw.words)} 个词（{'、'.join(kw.words[:8])}{'...' if len(kw.words) > 8 else ''}）")

    # 4. 点赞率阈值
    logger.info(f"点赞率阈值：{CONFIG.get('filter.like_rate_threshold')}  （口径：点赞/播放，模式 {CONFIG.get('filter.play_count_mode')}）")

    # 5. 自动点赞开关
    if CONFIG.get("auto_like.enabled", False):
        print()
        print("  " + "!" * 68)
        print("  !! 自动点赞已开启：存在抖音风控 / 限流 / 封号风险，风险由使用者自行承担 !!")
        print("  " + "!" * 68)
        logger.warning("自动点赞开关 = 开启")
    else:
        logger.info("自动点赞开关：关闭（仅识别分析，不会点击任何按钮）")

    logger.info(f"输出表格：{CONFIG.path('output.excel_path')}")
    logger.info(f"日志目录：{CONFIG.path('output.log_dir')}")


def process_one(win: WinInfo, screen: Screen, frames: list, logger):
    t0 = time.time()
    main_frame = frames[0]

    # ---------- OCR ----------
    t_ocr = time.time()
    lines_by_region = ocr_mod.ocr_frame(main_frame)
    t_ocr = time.time() - t_ocr
    all_lines = [ln for group in lines_by_region.values() for ln in group]
    all_text = " ".join(ln.text for ln in all_lines)

    caption_lines = lines_by_region.get("caption", [])
    profile_lines = lines_by_region.get("profile", [])
    rail_lines = lines_by_region.get("rail", [])
    comment_lines = lines_by_region.get("comments", [])

    # ---------- 解析 ----------
    prefixes = CONFIG.get("ocr.search_term_prefixes", [])
    heur = bool(CONFIG.get("ocr.search_term_heuristics", True))
    search_confirmed, search_guessed = extract_search_terms(caption_lines + all_lines, prefixes, heur)
    hashtags = extract_hashtags(all_lines)
    nickname = extract_nickname(profile_lines + all_lines)
    metrics = extract_metrics(rail_lines, all_lines, comment_lines)
    comments = extract_comments(comment_lines)

    # ---------- 视觉 ----------
    vres = vision_mod.analyze(frames, logger)

    # ---------- 判定 ----------
    decision = judge(vres, search_confirmed, search_guessed, hashtags, metrics, comments, all_text)

    # ---------- 输出 ----------
    video_link, profile_link = build_links(nickname)
    search_show = " / ".join(search_confirmed[:3]) or ("~" + search_guessed[0] if search_guessed else "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_path = ""
    if decision.is_candidate or (decision.is_spam and CONFIG.get("filter.record_spam_to_excel", True)):
        if CONFIG.get("capture.save_frame_on_hit", True):
            fdir = CONFIG.path("capture.save_frame_dir", "data/frames")
            os.makedirs(fdir, exist_ok=True)
            safe = "".join(ch for ch in (nickname or "unknown") if ch.isalnum() or ch in "._-")[:24] or "unknown"
            frame_path = os.path.join(fdir, f"{datetime.now():%Y%m%d_%H%M%S}_{safe}.jpg")
            ok, buf = cv2.imencode(".jpg", main_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                buf.tofile(frame_path)

    record = {
        "nickname": nickname,
        "video_link": video_link,
        "profile_link": profile_link,
        "search_terms": search_show,
        "hashtags": " ".join("#" + t for t in hashtags),
        "hit_keywords": "、".join(decision.hit_keywords),
        "reason": decision.reason_text,
        "play": _j(metrics.play),
        "like": _j(metrics.like),
        "comment": _j(metrics.comment),
        "like_rate": f"{decision.like_rate:.2%}" if decision.like_rate is not None else "",
        "grade": decision.grade,
        "ts": ts,
        "frame_path": frame_path,
    }

    # 全量 JSONL 日志：默认只在「合格视频」时记录；save_full_log=true 时才连被过滤的一起记
    if CONFIG.get("output.save_full_log", False):
        write_jsonl({
            "kind": "video",
            "nickname": nickname,
            "grade": decision.grade,
            "score": decision.score,
            "vision": vres.as_log(),
            "metrics": {"play": metrics.play, "like": metrics.like, "comment": metrics.comment,
                        "collect": metrics.collect, "share": metrics.share},
            "like_rate": decision.like_rate,
            "like_rate_note": decision.like_rate_note,
            "hashtags": hashtags,
            "search_confirmed": search_confirmed,
            "search_guessed": search_guessed,
            "hit_keywords": decision.hit_keywords,
            "spam_reasons": decision.spam_reasons,
            "reject_reasons": decision.reject_reasons,
            "reason": decision.reason_text,
            "ocr_text": all_text[:2000],
            "ocr_seconds": round(t_ocr, 2),
            "elapsed": round(time.time() - t0, 2),
        }, True)

    return {
        "record": record,
        "decision": decision,
        "vision": vres,
        "metrics": metrics,
        "hashtags": hashtags,
        "search_confirmed": search_confirmed,
        "comments": comments,
        "nickname": nickname,
        "candidate": decision.is_candidate,
    }


def run() -> int:
    global _STOP
    enable_dpi_awareness()
    signal.signal(signal.SIGINT, _on_sigint)

    logger = setup_logger(CONFIG.path("output.log_dir", "logs"), bool(CONFIG.get("output.console_verbose", True)))
    startup_checks(logger)

    writer = ExcelWriter(CONFIG.path("output.excel_path"), str(CONFIG.get("output.sheet_name", "候选达人")),
                         bool(CONFIG.get("output.excel_freeze_header", True)),
                         int(CONFIG.get("safety.dedup_memory", 400)))

    # ---------- 定位窗口 ----------
    win = None
    resolved_note = ""
    for attempt in range(3):
        win, resolved_note = resolve_douyin_window(
            CONFIG.get("window.title_keywords", []),
            CONFIG.get("window.process_name", ""),
            bool(CONFIG.get("window.auto_restore_hidden", True)),
        )
        if win:
            break
        time.sleep(1.0)

    if win and resolved_note == "restored_from_tray":
        logger.info("检测到抖音客户端缩在系统托盘，已自动把窗口调回桌面")
        restored_from_tray = True
    else:
        restored_from_tray = False

    if not win:
        # ---------- 客户端没在运行 → 自动启动 ----------
        if (resolved_note == "not_running"
                and CONFIG.get("window.launch_if_not_running", True)):
            exe = find_douyin_exe(CONFIG.get("window.executable", ""))
            if exe:
                logger.info(f"未检测到抖音客户端，自动启动：{exe}")
                if launch_douyin(exe):
                    deadline = time.time() + 75
                    tray_tried = False
                    while time.time() < deadline and not _STOP:
                        time.sleep(3.0)
                        win, resolved_note = resolve_douyin_window(
                            CONFIG.get("window.title_keywords", []),
                            CONFIG.get("window.process_name", ""),
                            bool(CONFIG.get("window.auto_restore_hidden", True)),
                        )
                        if win:
                            logger.info("抖音客户端已启动并锁定窗口")
                            break
                        # 该客户端启动后常驻托盘；等一半时间还没窗口就尝试点托盘图标唤出
                        if not tray_tried and time.time() > deadline - 45:
                            tray_tried = True
                            note = open_douyin_via_tray()
                            logger.info(f"托盘唤起：{note}")
                    if not win:
                        logger.warning("等待抖音客户端启动超时（75 秒）")
                else:
                    logger.error(f"启动抖音客户端失败：{exe}")
            else:
                logger.warning("未找到抖音客户端安装路径，无法自动启动"
                               "（可在 config.yaml 的 window.executable 里手动填 douyin.exe 完整路径）")

    if not win:
        diag = diagnose_windows(CONFIG.get("window.process_name", "douyin.exe"))
        diag_txt = "\n".join("  · " + d for d in diag) if diag else "  （未检测到相关窗口）"
        hint = {
            "hidden_in_tray": "客户端缩在系统托盘里。可以点击托盘图标把窗口调出来，\n"
                              "或把 config.yaml 里 window.auto_restore_hidden 设为 true 让脚本自动调出。",
            "restore_failed": "尝试调出托盘窗口失败。请手动点击托盘图标打开抖音窗口。",
            "not_running": "没有检测到抖音电脑客户端进程，且自动启动未成功，请先手动启动客户端。",
        }.get(resolved_note, "请确认抖音电脑客户端已经打开且窗口可见。")
        msg = (f"没有找到可用的抖音客户端窗口。\n\n{hint}\n\n"
               f"当前检测到的相关窗口：\n{diag_txt}\n\n"
               "处理完后重新运行本工作流。")
        logger.error(f"未找到抖音窗口（{resolved_note}），暂停运行")
        for d in diag:
            logger.error("  " + d)
        _popup("抖音达人筛选工作流 · 需要人工处理", msg)
        return 2

    logger.info(f"已锁定窗口：{win.title}  ({win.width}x{win.height}) @ ({win.rect[0]},{win.rect[1]})")

    if _OPTS["check"]:
        print(banner("自检模式：窗口定位成功，配置与依赖均正常。\n去掉 --check 参数即可开始正式运行。"))
        return 0

    screen = Screen()
    human = Human(pyautogui.size().width, pyautogui.size().height)

    # ---------- 切到推荐页 ----------
    if CONFIG.get("window.auto_switch_feed", True):
        tab = CONFIG.get("window.feed_tab", "推荐")
        targets = [str(t) for t in tab] if isinstance(tab, (list, tuple)) else [str(tab)]
        if CONFIG.get("window.keep_foreground", True):
            activate(win.hwnd)
        time.sleep(1.0)
        try:
            okc, notec = human.click_text(win, screen, targets)
            if okc:
                logger.info(f"导航：{notec}")
            else:
                logger.warning(f"切换推荐页未成功（不影响运行）：{notec}")
        except Exception as exc:
            logger.warning(f"切换推荐页异常（不影响运行）：{exc}")

    hwnd = win.hwnd
    last_fp = ""
    same_count = 0
    idx = 0
    kept = 0
    spam_n = 0
    reject_n = 0
    liked = 0
    started = time.time()
    max_min = int(CONFIG.get("safety.max_runtime_minutes", 0) or 0)

    print(banner("开始运行。随时可以：改 keywords.txt / config.yaml（实时生效） | Ctrl+C 退出 | 鼠标甩到左上角紧急制动"))

    try:
        while not _STOP:
            # ---------- 0. 热更新配置 ----------
            changed = CONFIG.refresh()
            if changed:
                logger.info(f"🔄 检测到配置变更并已热更新：{'、'.join(changed)}")

            # ---------- 1. 运行时长 ----------
            if max_min and (time.time() - started) / 60.0 >= max_min:
                logger.info(f"已达到设定的最长运行时长 {max_min} 分钟，正常结束")
                break

            # ---------- 2. 窗口健康检查 ----------
            if not is_alive(hwnd):
                logger.error("抖音窗口已关闭")
                _popup("抖音达人筛选工作流 · 已暂停",
                       "抖音窗口已经关闭，工作流已暂停。\n\n重新打开客户端后可重新运行脚本。")
                break
            if is_minimized(hwnd):
                if CONFIG.get("window.restore_if_minimized", False):
                    logger.info("抖音窗口被最小化，按配置自动还原")
                    show_no_activate(hwnd, activate=True)
                    time.sleep(1.2)
                    continue
                msg = ("抖音窗口被最小化了，工作流已暂停。\n\n"
                       "请把抖音窗口还原到桌面可见状态，然后点击「确定」继续运行。")
                logger.warning("抖音窗口被最小化，暂停并弹窗提醒")
                _popup("抖音达人筛选工作流 · 已暂停", msg)
                if _OPTS["no_alert"] or _STOP:
                    break
                # 用户确认后自动还原，避免反复弹窗
                show_no_activate(hwnd, activate=True)
                time.sleep(1.2)
                continue
            rect = refresh_rect(hwnd)
            if rect:
                win = WinInfo(hwnd, win.title, rect)

            # ---------- 3. 确保前台 ----------
            if CONFIG.get("window.keep_foreground", True):
                if not activate(hwnd):
                    logger.warning("窗口无法切到前台，稍后重试")
                    time.sleep(1.5)
                    continue

            # ---------- 4. 抽帧 ----------
            frames = []
            try:
                f1 = grab_window(win, screen)
            except Exception as exc:
                logger.warning(f"截屏失败：{exc}")
                time.sleep(1.0)
                continue
            if is_bad_frame(f1):
                time.sleep(1.0)
                continue
            frames.append(f1)

            n_frames = max(1, int(CONFIG.get("vision.frames_per_video", 1) or 1))
            for _ in range(n_frames - 1):
                time.sleep(rnd(CONFIG.get("capture.sample_interval", [0.8, 1.4])))
                try:
                    fx = grab_window(win, screen)
                    if not is_bad_frame(fx):
                        frames.append(fx)
                except Exception:
                    pass

            # ---------- 5. 判断是否换了视频 ----------
            fp = frame_fingerprint(frames[-1])
            dist = hamming_hex(last_fp, fp)
            if last_fp and dist <= 3:
                same_count += 1
                if same_count >= int(CONFIG.get("safety.same_frame_retry", 3)):
                    logger.warning("画面连续多次未变化，换滚轮方式重试")
                    human.scroll_next(win, method="wheel")
                    same_count = 0
                    time.sleep(1.0)
                    continue
            else:
                same_count = 0
            last_fp = fp

            # ---------- 6. 分析 ----------
            idx += 1
            try:
                result = process_one(win, screen, frames, logger)
            except Exception as exc:
                logger.exception(f"处理异常：{exc}")
                time.sleep(1.0)
                continue

            decision = result["decision"]
            nickname = result["nickname"]

            # ---------- 7. 控制台输出 ----------
            if CONFIG.get("output.console_verbose", True):
                print(console_line(idx, nickname, result["vision"], decision, result["metrics"],
                                   result["hashtags"], result["search_confirmed"], decision.hit_keywords))

            # ---------- 8. 落表 ----------
            write_it = False
            if decision.is_candidate:
                write_it = True
                kept += 1
            elif decision.grade == GRADE_SPAM and CONFIG.get("filter.record_spam_to_excel", True):
                write_it = True
                spam_n += 1
            elif decision.grade == GRADE_REJECT:
                reject_n += 1
                if CONFIG.get("filter.record_rejected_to_excel", False):
                    write_it = True

            if write_it:
                wrote = writer.append(result["record"], nickname, fp[:16])
                if not wrote:
                    logger.debug(f"重复达人，跳过写入：{nickname}")

            # ---------- 9. 自动点赞 ----------
            if (CONFIG.get("auto_like.enabled", False)
                    and decision.grade in (GRADE_HIGH, GRADE_NORMAL)
                    and not decision.is_spam):
                if human.can_like():
                    ok, note = human.click_like(win, frames[0])
                    if ok:
                        liked += 1
                        logger.info(f"👍 已点赞（{note}） — {nickname or '未知达人'}  [本场第 {liked} 次]")
                        time.sleep(rnd(CONFIG.get("auto_like.delay_after_like", [3, 6])))
                    else:
                        logger.info(f"自动点赞跳过：{note}")

            if writer.last_warning():
                logger.warning(writer.last_warning())

            # ---------- 10. 拟人停顿 + 滚动 ----------
            time.sleep(rnd(CONFIG.get("rhythm.dwell_seconds", [2, 5])))

            if _OPTS["once"]:
                logger.info("--once 参数生效：已处理 1 条视频，正常退出")
                break

            human.maybe_rest()
            human.scroll_next(win)

    except KeyboardInterrupt:
        pass
    finally:
        screen.close()
        # 如果抖音窗口原本是缩在托盘里的，退出时帮你还原回去
        if restored_from_tray and is_alive(hwnd) and not is_minimized(hwnd):
            try:
                hide_window(hwnd)
                print("（抖音窗口原本缩在托盘，已帮你还原）")
            except Exception:
                pass

    mins = (time.time() - started) / 60.0
    print(banner(
        f"运行结束\n"
        f"  共识别视频：{idx} 条\n"
        f"  候选达人（高匹配/普通）：{kept} 条\n"
        f"  刷量号：{spam_n} 条\n"
        f"  未达标：{reject_n} 条\n"
        f"  自动点赞：{liked} 次\n"
        f"  写入表格累计：{writer.count} 行\n"
        f"  运行时长：{mins:.1f} 分钟\n"
        f"  结果表格：{CONFIG.path('output.excel_path')}\n"
        f"  详细日志：{CONFIG.path('output.log_dir')}"
    ))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="抖音美瞳达人筛选工作流",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="只做启动自检与窗口定位，不进入识别循环（推荐首次使用）")
    parser.add_argument("--once", action="store_true",
                        help="只识别 1 条视频就退出，用于试跑观察效果")
    parser.add_argument("--no-alert", action="store_true",
                        help="禁用弹窗提醒（窗口异常时改为控制台输出后退出）")
    args = parser.parse_args()

    _OPTS.update(check=args.check, once=args.once, no_alert=args.no_alert)
    sys.exit(run())
