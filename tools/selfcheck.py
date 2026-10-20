# -*- coding: utf-8 -*-
r"""离线自检：不动抖音、不刷视频，验证整条链路是否正常。

运行：  tools\selfcheck.py（用项目自带的 Python 环境执行）

依次验证：
  1) 依赖是否齐全
  2) 能否定位到抖音窗口
  3) 能否截取抖音窗口画面（存到 data/preview/）
  4) 文字识别能否读出画面上的字
  5) 解析 / 判定 / Excel 写入 是否正常（用合成画面模拟一条抖音视频）
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.config import CONFIG, ROOT_DIR


def step(n: int, title: str) -> None:
    print(f"\n{'─' * 70}\n[{n}] {title}\n{'─' * 70}")


def main() -> int:
    print("抖音美瞳达人筛选工作流 · 离线自检")

    # ---------------- 1. 依赖 ----------------
    step(1, "依赖检查")
    missing = []
    for mod in ("yaml", "requests", "numpy", "cv2", "PIL", "mss", "pyautogui", "openpyxl"):
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except Exception as exc:
            print(f"  ✗ {mod}  -> {exc}")
            missing.append(mod)
    from src import ocr as ocr_mod
    eng = ocr_mod.engine_status()
    print(f"  ✓ OCR 引擎：{eng}" if eng != "vision" else f"  ! OCR 引擎：{eng}（本地 OCR 不可用，将用大模型读字）")
    if missing:
        print("\n缺少依赖，请先按 README 安装。")
        return 1

    from src import vision as vision_mod
    from src.winutil import (Screen, diagnose_windows, enable_dpi_awareness,
                             find_douyin_window, grab_window, is_minimized)

    enable_dpi_awareness()

    # ---------------- 2. 窗口定位 ----------------
    step(2, "抖音窗口定位")
    win = find_douyin_window(CONFIG.get("window.title_keywords", []), CONFIG.get("window.process_name", ""))
    diag = diagnose_windows(CONFIG.get("window.process_name", "douyin.exe"))
    if not win:
        print("  ✗ 未找到抖音窗口。")
        print(f"    检测到的 douyin.exe 窗口（共 {len(diag)} 个）：")
        if diag:
            for d in diag:
                print(f"      {d}")
        else:
            print("      无 —— 请打开抖音电脑客户端，或把窗口从托盘还原出来")
        print("    （后面的步骤会用合成画面继续验证，不影响结论）")
    else:
        print(f"  ✓ 命中窗口：{win.title!r}")
        print(f"    位置 {win.rect}  尺寸 {win.width}x{win.height}")
        print(f"    最小化：{'是（请还原窗口）' if is_minimized(win.hwnd) else '否'}")
    print(f"  · 人脸检测后端：{vision_mod.face_backend()}")

    # ---------------- 3. 截屏 ----------------
    frame = None
    if win and not is_minimized(win.hwnd):
        step(3, "窗口截屏")
        try:
            screen = Screen()
            frame = grab_window(win, screen)
            screen.close()
            outdir = os.path.join(ROOT_DIR, "data", "preview")
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, "douyin_window.jpg")
            cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(out)
            print(f"  ✓ 截屏成功 {frame.shape[1]}x{frame.shape[0]}")
            print(f"    已保存预览：{out}")
        except Exception as exc:
            print(f"  ✗ 截屏失败：{exc}")
    else:
        step(3, "窗口截屏（跳过：窗口未就绪）")

    # ---------------- 4. 文字识别 ----------------
    if frame is not None:
        step(4, "文字识别（真实画面）")
        t0 = time.time()
        lines_by_region = ocr_mod.ocr_frame(frame)
        total = sum(len(v) for v in lines_by_region.values())
        print(f"  ✓ 识别到 {total} 段文字，耗时 {time.time() - t0:.2f}s")
        for name, lines in lines_by_region.items():
            if not lines:
                continue
            print(f"    · 区域 {name}:")
            for ln in lines[:8]:
                print(f"        {ln.text}   (置信度 {ln.score:.2f})")
        if total == 0:
            print("    ! 没有识别到文字。可能原因：抖音窗口被遮挡 / 当前是纯视频画面 / OCR 裁剪区域需要按你的客户端微调。")

        from src.parser import (extract_hashtags, extract_metrics, extract_nickname,
                                extract_search_terms)
        all_lines = [ln for g in lines_by_region.values() for ln in g]
        conf, guess = extract_search_terms(all_lines, CONFIG.get("ocr.search_term_prefixes", []),
                                           bool(CONFIG.get("ocr.search_term_heuristics", True)))
        print(f"    → 看后搜词条(确定)：{conf or '无'}")
        print(f"    → 看后搜词条(推测)：{guess or '无'}")
        print(f"    → 话题标签：{extract_hashtags(all_lines) or '无'}")
        print(f"    → 达人昵称：{extract_nickname(lines_by_region.get('profile', []) + all_lines) or '无'}")
        m = extract_metrics(lines_by_region.get("rail", []), all_lines, lines_by_region.get("comments", []))
        print(f"    → 互动数据：点赞={m.like} 评论={m.comment} 收藏={m.collect} 分享={m.share} 播放={m.play}")
    else:
        step(4, "文字识别（跳过：无画面）")

    # ---------------- 5. 合成画面 → 解析/判定/写表 ----------------
    step(5, "解析 / 判定 / Excel 写入（合成抖音画面）")
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int):
        for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if os.path.isfile(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    W, H = 1280, 800
    img = Image.new("RGB", (W, H), (24, 22, 28))
    d = ImageDraw.Draw(img)
    # 模拟人脸区域（放一个圆润的肤色块 + 双眼，便于本地预筛/本地降级有个东西可看）
    d.ellipse([420, 160, 860, 660], fill=(232, 196, 172))
    d.ellipse([520, 330, 590, 380], fill=(60, 45, 40))
    d.ellipse([690, 330, 760, 380], fill=(60, 45, 40))
    d.ellipse([600, 470, 680, 500], fill=(190, 130, 130))
    d.text((40, 30), "小鹿美妆日记", font=load_font(30), fill=(255, 255, 255))
    d.text((60, 620), "新入的微混血美瞳太绝了 #美瞳 #妆容分享 #日抛", font=load_font(30), fill=(255, 255, 255))
    d.text((60, 680), "看后搜：可啦啦月抛试戴", font=load_font(28), fill=(255, 220, 120))
    for i, txt in enumerate(["1.8万", "326", "902", "158"]):
        d.text((1130, 300 + i * 70), txt, font=load_font(26), fill=(230, 230, 230))
    d.text((1050, 596), "评论", font=load_font(22), fill=(180, 180, 180))
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    outdir = os.path.join(ROOT_DIR, "data", "preview")
    os.makedirs(outdir, exist_ok=True)
    syn = os.path.join(outdir, "synthetic_frame.jpg")
    cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(syn)
    print(f"  · 合成画面已保存：{syn}")

    lines_by_region = ocr_mod.ocr_frame(bgr)
    all_lines = [ln for g in lines_by_region.values() for ln in g]
    print(f"  ✓ OCR 读到 {len(all_lines)} 段：")
    for ln in all_lines[:12]:
        print(f"      {ln.text}")

    from src.judge import judge
    from src.parser import (extract_comments, extract_hashtags, extract_metrics,
                            extract_nickname, extract_search_terms)
    conf, guess = extract_search_terms(all_lines, CONFIG.get("ocr.search_term_prefixes", []),
                                       bool(CONFIG.get("ocr.search_term_heuristics", True)))
    tags = extract_hashtags(all_lines)
    nick = extract_nickname(lines_by_region.get("profile", []) + all_lines)
    m = extract_metrics(lines_by_region.get("rail", []), all_lines, lines_by_region.get("comments", []))
    all_text = " ".join(l.text for l in all_lines)

    vres = vision_mod.local_analyze(bgr)
    dec = judge(vres, conf, guess, tags, m, [], all_text)
    print(f"  ✓ 解析结果：昵称={nick or '无'} 话题={tags} 看后搜={conf or guess or '无'}")
    print(f"  ✓ 数据：点赞={m.like} 评论={m.comment} 收藏={m.collect} 分享={m.share} 播放={m.play}")
    print(f"  ✓ 本地视觉：{vres.as_log()}")
    print(f"  ✓ 判定：等级={dec.grade} 综合分={dec.score} 命中={dec.hit_keywords} 理由={dec.reason_text}")
    print("     （合成图是手绘色块，YuNet 检不出人脸 → 判为不达标，属于预期行为；")
    print("      下面换成模拟的 AI 结果，验证命中 / 高匹配 / 写表的完整流程）")

    # ---------------- 5b. 模拟 AI 视觉通过 → 验证命中与等级 ----------------
    step("5b", "关键词命中 / 等级判定 / Excel 落表（模拟 AI 结果）")
    from src.config import KeywordLib
    from src.vision import VisionResult

    old_kw = CONFIG._kw
    CONFIG._kw = KeywordLib(path="<自检临时关键词库>", words=["美瞳", "微混血", "可啦啦"])
    fake = VisionResult(
        ok=True, source="api", has_person=True, face_visible=True, eyes_visible=True,
        eye_count=2, face_ratio=0.18, eye_area_quality=86.0, beauty_score=88.0,
        lens_suitable=87.0, content_type="真人出镜",
        reason="眼部近景清晰、眼妆精致、瞳孔可见，适合佩戴美瞳拍种草视频", confidence=0.9,
    )
    dec2 = judge(fake, conf, guess, tags, m, [], all_text)
    hit_rate = f"{dec2.like_rate:.2%}" if dec2.like_rate is not None else "未计算"
    print(f"  ✓ 临时关键词库：美瞳 / 微混血 / 可啦啦")
    print(f"  ✓ 命中关键词：{dec2.hit_keywords or '无'}")
    print(f"  ✓ 等级：{dec2.grade}   综合分：{dec2.score}   点赞率：{hit_rate}")
    print(f"  ✓ 判定理由：{dec2.reason_text}")
    CONFIG._kw = old_kw

    from src.excel_out import ExcelWriter
    test_path = os.path.join(ROOT_DIR, "data", "自检输出.xlsx")
    if os.path.isfile(test_path):
        try:
            os.remove(test_path)
        except OSError:
            pass
    w = ExcelWriter(test_path, "候选达人")
    nick_a = nick or "自检测试达人"
    nick_b = (nick + "(模拟AI通过)") if nick else "自检测试达人2"
    # 第一行：上一条真实链路（本地视觉未通过）
    w.append({
        "nickname": nick or "自检测试达人", "video_link": "", "profile_link": "",
        "search_terms": "", "hashtags": " ".join("#" + t for t in tags),
        "hit_keywords": "", "reason": dec.reason_text,
        "play": m.play, "like": m.like, "comment": m.comment,
        "like_rate": "", "grade": dec.grade, "frame_path": syn,
    }, nick or "自检测试达人", "selftest-a")
    # 第二行：模拟 AI 通过 + 命中关键词
    w.append({
        "nickname": nick_b,
        "video_link": "https://www.douyin.com/search/%E5%B0%8F%E9%B9%BF%E7%BE%8E%E5%A6%86%E6%97%A5%E8%AE%B0",
        "profile_link": "https://www.douyin.com/search/%E5%B0%8F%E9%B9%BF%E7%BE%8E%E5%A6%86%E6%97%A5%E8%AE%B0",
        "search_terms": " / ".join(conf[:2]), "hashtags": " ".join("#" + t for t in tags),
        "hit_keywords": "、".join(dec2.hit_keywords), "reason": dec2.reason_text,
        "play": m.play, "like": m.like, "comment": m.comment,
        "like_rate": f"{dec2.like_rate:.2%}" if dec2.like_rate is not None else "",
        "grade": dec2.grade, "frame_path": syn,
    }, nick_b, "selftest-b")
    print(f"  ✓ Excel 写入成功（2 行示例）：{test_path}")
    if vision_mod.face_backend() == "statistics":
        print("  ! 未加载到人脸检测模型，本地预筛精度会下降。")
        print(f"    请确认模型文件存在：{CONFIG.get('vision.prefilter.face_model')}")

    print(f"\n{'=' * 70}\n自检完成。以上链路全部走通后，直接双击 run.bat 即可开始正式运行。\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
