# -*- coding: utf-8 -*-
"""AI 视觉识别：判断达人眼部条件 / 颜值 / 是否适合美瞳种草，并识别视频是否为美瞳内容。

两条链路：
  1) 主链路：OpenAI 兼容的多模态接口（默认 DeepSeek deepseek-flash；
     也支持 qwen-vl-max / GLM-4V / 豆包 / GPT-4o 等）
  2) 降级链路：纯本地 OpenCV 启发式（人脸+眼睛检测、清晰度、遮挡度）
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from .config import CONFIG, ROOT_DIR

# ------------------------------------------------------------------ 数据结构
@dataclass
class VisionResult:
    ok: bool = False
    source: str = "none"          # api / local / prefilter / none
    has_person: bool = False
    face_visible: bool = False
    eyes_visible: bool = False
    eye_count: int = 0
    face_ratio: float = 0.0
    eye_area_quality: float = 0.0
    beauty_score: float = 0.0
    lens_suitable: float = 0.0    # 0~100，核心分
    content_type: str = ""
    is_ad_or_live: bool = False
    occlusion: float = 1.0        # 0=无遮挡 1=完全遮挡
    lens_content: bool = False    # 画面里是否出现美瞳相关内容（文字/贴纸/字幕）
    lens_brand_hits: List[str] = field(default_factory=list)   # 识别到的美瞳品牌/品类词
    theme: str = ""               # 视频主题
    reason: str = ""
    confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_log(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "source": self.source, "lens_suitable": round(self.lens_suitable, 1),
            "eyes_visible": self.eyes_visible, "face_visible": self.face_visible,
            "content_type": self.content_type, "is_ad_or_live": self.is_ad_or_live,
            "lens_content": self.lens_content, "lens_brand_hits": self.lens_brand_hits,
            "reason": self.reason, "error": self.error,
        }


# ------------------------------------------------------------------ 提示词
SYSTEM_PROMPT = (
    "你是美瞳（彩色隐形眼镜）品牌方的达人筛选专家。"
    "你的任务是看抖音视频截图，做两件事："
    "① 判断画面里的达人是否适合拍摄美瞳种草视频；"
    "② 判断这条视频本身是否和美瞳相关（画面里的文案、贴纸、字幕、商品上有没有出现美瞳、彩瞳、"
    "隐形眼镜、品牌名、日抛月抛、瞳色/直径/花色等美瞳相关字眼）。"
    "美瞳种草的核心要求是：真人出镜、面部清晰、眼睛看得清（能分辨瞳孔/虹膜/妆容细节）、"
    "上镜好看、有眼部特写或近景。"
    "如果画面没有真人、面部被大面积遮挡（手、口罩、贴纸、特效、头发）、眼睛看不清、"
    "或者画面是广告/直播/纯图文/宠物/风景，则不适合。"
)

USER_PROMPT = """请分析这张抖音视频截图。只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。

字段说明：
{
  "has_person": 布尔,               // 画面中是否有真人
  "face_visible": 布尔,             // 面部是否清晰可见（未被遮挡、未打码、非背影）
  "eyes_visible": 布尔,             // 双眼是否清晰可见，能否看清瞳孔与眼妆细节
  "eye_count": 整数,                // 能清晰看到的眼睛数量：0/1/2
  "face_ratio": 浮点,               // 人脸面积占画面比例，0~1
  "eye_area_quality": 浮点,         // 眼部区域条件打分 0~100（清晰度、大小、是否适合展示美瞳）
  "beauty_score": 浮点,             // 颜值/上镜度打分 0~100
  "makeup_style": 字符串,           // 妆容风格，如"韩系裸妆""微混血""浓妆""素颜"
  "content_type": 字符串,           // 内容类型：真人出镜/穿搭/口播/剧情/图文/广告/直播/宠物/风景/其他
  "is_ad_or_live": 布尔,            // 是否为广告或直播画面
  "lens_content": 布尔,             // 画面中的文字/贴纸/字幕/商品上，是否出现美瞳相关内容
                                    //   （美瞳、彩瞳、彩色隐形眼镜、隐形眼镜、瞳片、品牌名、
                                    //     日抛/月抛/硅水凝胶、瞳色/直径/花色 等）
  "lens_brand_hits": [字符串],      // 上面识别到的具体美瞳相关词或品牌名，逐个列出；没有就给 []
  "theme": 字符串,                  // 视频主题，如"美瞳种草""美妆""穿搭""剧情""口播""其他"
  "lens_suitable": 浮点,            // 综合打分 0~100：该达人是否适合佩戴美瞳拍摄种草视频
  "reason": 字符串,                 // 一句话中文理由，40字以内
  "confidence": 浮点                // 你的判断置信度 0~1
}"""


# ------------------------------------------------------------------ 图片编码
def encode_jpeg(img: np.ndarray, max_width: int = 1280, quality: int = 88) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def _to_data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


# ------------------------------------------------------------------ 人脸检测
@dataclass
class FaceInfo:
    box: Tuple[int, int, int, int]          # x, y, w, h
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    score: float = 1.0


_YUNET = None
_YUNET_FAILED = False
_HAAR = None
_HAAR_FAILED = False


def _yunet_model_path() -> str:
    p = CONFIG.get("vision.prefilter.face_model", "models/face_detection_yunet_2023mar.onnx")
    return p if os.path.isabs(p) else os.path.join(ROOT_DIR, p)


def _ensure_yunet_model() -> Optional[str]:
    path = _yunet_model_path()
    if os.path.isfile(path):
        return path
    if not CONFIG.get("vision.prefilter.auto_download_model", True):
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urls = [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ]
    for u in urls:
        try:
            import urllib.request
            urllib.request.urlretrieve(u, path)
            if os.path.getsize(path) > 10000:
                return path
        except Exception:
            continue
    return None


def _get_yunet():
    """YuNet 人脸检测器（OpenCV 5.x 移除了 Haar，这是新的首选方案）。"""
    global _YUNET, _YUNET_FAILED
    if _YUNET is not None or _YUNET_FAILED:
        return _YUNET
    if not hasattr(cv2, "FaceDetectorYN"):
        _YUNET_FAILED = True
        return None
    path = _ensure_yunet_model()
    if not path:
        _YUNET_FAILED = True
        return None
    try:
        _YUNET = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.65, 0.3, 200)
    except Exception as exc:
        print(f"[视觉] YuNet 初始化失败，降级到其他方案：{exc}")
        _YUNET_FAILED = True
    return _YUNET


def _get_haar():
    """OpenCV 4.x 的 Haar 级联（老环境兜底）。"""
    global _HAAR, _HAAR_FAILED
    if _HAAR is not None or _HAAR_FAILED:
        return _HAAR
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        _HAAR_FAILED = True
        return None
    try:
        base = cv2.data.haarcascades
        face = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
        if face.empty():
            _HAAR_FAILED = True
            return None
        _HAAR = face
    except Exception:
        _HAAR_FAILED = True
    return _HAAR


def face_backend() -> str:
    if _get_yunet() is not None:
        return "yunet"
    if _get_haar() is not None:
        return "haar"
    return "statistics"


def detect_faces(img: np.ndarray) -> List[FaceInfo]:
    h, w = img.shape[:2]
    out: List[FaceInfo] = []

    det = _get_yunet()
    if det is not None:
        try:
            det.setInputSize((w, h))
            _ret, faces = det.detect(img)
            if faces is not None:
                for f in faces:
                    x, y, fw, fh = [int(v) for v in f[:4]]
                    pts = []
                    lm = f[4:14]
                    for i in range(0, len(lm), 2):
                        pts.append((float(lm[i]), float(lm[i + 1])))
                    out.append(FaceInfo((x, y, fw, fh), pts, float(f[14] if len(f) > 14 else 1.0)))
        except Exception:
            pass
        if out:
            return out

    haar = _get_haar()
    if haar is not None:
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        mins = max(24, int(min(h, w) * 0.06))
        for (x, y, fw, fh) in haar.detectMultiScale(gray, 1.12, 5, minSize=(mins, mins)):
            out.append(FaceInfo((int(x), int(y), int(fw), int(fh))))
    return out


# ------------------------------------------------------------------ 画面统计
def _skin_ratio(img: np.ndarray) -> float:
    """粗估肤色占比，用于没有人脸模型时的兜底判断。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 70], np.uint8), np.array([25, 180, 255], np.uint8))
    mask |= cv2.inRange(hsv, np.array([160, 30, 70], np.uint8), np.array([180, 180, 255], np.uint8))
    return float(np.count_nonzero(mask)) / float(hsv.shape[0] * hsv.shape[1])


# ------------------------------------------------------------------ 预筛
@dataclass
class PrefilterOut:
    passed: bool
    reason: str = ""
    face_px: int = 0
    face_ratio: float = 0.0
    eye_sharpness: float = 0.0
    backend: str = ""


def prefilter(img: np.ndarray) -> PrefilterOut:
    """本地快速预筛。

    strict=true：人脸太小/没脸/眼睛糊 → 拦截（省钱，但容易误杀）。
    strict=false（默认）：只拦"整体模糊"，其余全部交给 AI 判断 —— 本地
    人脸检测对侧脸/远镜/剧情镜头漏检率高，拦掉会导致 AI 没机会看。
    """
    cfg = CONFIG.get("vision.prefilter", {})
    backend = face_backend()
    if not cfg.get("enabled", True):
        return PrefilterOut(True, "预筛已关闭", backend=backend)
    strict = bool(cfg.get("strict", False))

    h, w = img.shape[:2]
    faces = detect_faces(img)

    # ---- 方案 C：人脸模型全不可用，退回图像统计 ----
    if not faces and backend == "statistics":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        skin = _skin_ratio(img)
        if sharp < float(cfg.get("min_eye_sharpness", 45)) * 0.6:
            return PrefilterOut(False, f"画面整体模糊(sharpness={sharp:.0f})", 0, 0.0, sharp, backend)
        if strict and skin < 0.02:
            return PrefilterOut(False, "画面中未发现人像肤色", 0, 0.0, sharp, backend)
        return PrefilterOut(True, "统计法预筛通过（未装人脸模型，建议放好 YuNet 模型）",
                            0, skin, sharp, backend)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    whole_sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if whole_sharp < float(cfg.get("min_eye_sharpness", 45)) * 0.6:
        return PrefilterOut(False, f"画面整体模糊({whole_sharp:.0f})", 0, 0.0, whole_sharp, backend)

    if not faces:
        if strict:
            return PrefilterOut(False, "未检测到人脸", 0, 0.0, whole_sharp, backend)
        return PrefilterOut(True, "本地未检出人脸，交给AI判断", 0, 0.0, whole_sharp, backend)

    face = max(faces, key=lambda f: f.box[2] * f.box[3])
    fx, fy, fw, fh = face.box
    face_px = int(min(fw, fh))
    face_ratio = float(fw * fh) / float(w * h)

    # 有 5 点关键点 → 直接框出双眼区域；否则退化到脸部上 40%
    if len(face.landmarks) >= 2:
        exs = [p[0] for p in face.landmarks[:2]]
        eys = [p[1] for p in face.landmarks[:2]]
        pad = max(8, int(fw * 0.18))
        ex0, ex1 = int(min(exs)) - pad, int(max(exs)) + pad
        ey0, ey1 = int(min(eys)) - pad, int(max(eys)) + pad
    else:
        ex0, ex1 = fx, fx + fw
        ey0, ey1 = fy + int(fh * 0.22), fy + int(fh * 0.62)
    ex0, ey0 = max(0, ex0), max(0, ey0)
    ex1, ey1 = min(w, max(ex1, ex0 + 2)), min(h, max(ey1, ey0 + 2))
    eye_roi = gray[ey0:ey1, ex0:ex1]
    sharpness = float(cv2.Laplacian(eye_roi, cv2.CV_64F).var()) if eye_roi.size > 0 else 0.0

    min_px = int(cfg.get("min_face_px", 140)) if strict else min(70, int(cfg.get("min_face_px", 140)))
    min_ratio = float(cfg.get("min_face_ratio", 0.02)) if strict else min(0.008, float(cfg.get("min_face_ratio", 0.02)))
    if face_px < min_px:
        if not strict and face_px >= 40:
            return PrefilterOut(True, f"人脸偏小({face_px}px)，交给AI判断", face_px, face_ratio, sharpness, backend)
        return PrefilterOut(False, f"人脸过小({face_px}px)，不是特写镜头", face_px, face_ratio, sharpness, backend)
    if face_ratio < min_ratio:
        return PrefilterOut(False, "人脸占画面比例过低", face_px, face_ratio, sharpness, backend)
    if sharpness < float(cfg.get("min_eye_sharpness", 45)):
        return PrefilterOut(False, f"眼部区域模糊({sharpness:.0f})，看不清美瞳效果", face_px, face_ratio, sharpness, backend)

    return PrefilterOut(True, "预筛通过", face_px, face_ratio, sharpness, backend)


# ------------------------------------------------------------------ 本地降级
def local_analyze(img: np.ndarray) -> VisionResult:
    """纯本地启发式：只能回答「眼睛是否看得清」，回答不了「颜值高低」。

    建议仅作为 API 不可用时的兜底；正式筛选请配置 vision.api_key。
    """
    pf = prefilter(img)
    res = VisionResult(source="local")
    res.face_ratio = round(pf.face_ratio, 4)
    res.eye_area_quality = round(min(100.0, pf.eye_sharpness / 3.0), 1)
    res.has_person = pf.face_px > 0 or pf.backend == "statistics"
    res.face_visible = pf.passed
    res.eyes_visible = pf.passed

    if not pf.passed:
        res.ok = True
        res.lens_suitable = 0.0
        res.beauty_score = 0.0
        res.reason = f"本地判定不通过：{pf.reason}"
        res.confidence = 0.45
        res.source = "prefilter"
        return res

    face_ratio_score = min(1.0, pf.face_ratio / 0.12) * 100 if pf.face_ratio else 55.0
    sharp_score = min(100.0, pf.eye_sharpness / 6.0)
    size_score = min(100.0, pf.face_px / 3.0) if pf.face_px else 55.0
    quality = 0.40 * face_ratio_score + 0.35 * sharp_score + 0.25 * size_score
    res.eye_area_quality = round(quality, 1)
    # 本地模式给保守分，避免误判成"高匹配"
    res.lens_suitable = round(min(78.0, 42.0 + quality * 0.36), 1)
    res.beauty_score = res.lens_suitable
    res.ok = True
    res.content_type = "真人出镜(本地判定)"
    res.reason = (f"本地判定：人脸占比{pf.face_ratio:.1%}、眼部清晰度{pf.eye_sharpness:.0f}"
                  f"（检测方式 {pf.backend}，未使用AI模型，仅供参考）")
    res.confidence = 0.45
    return res


# ------------------------------------------------------------------ API 调用
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "是", "真")


def api_analyze(frames: List[np.ndarray]) -> VisionResult:
    cfg = CONFIG.get("vision", {})
    key = (cfg.get("api_key") or "").strip()
    base = (cfg.get("base_url") or "").strip().rstrip("/")
    model = (cfg.get("model") or "").strip()
    if not key or not base or not model:
        return VisionResult(ok=False, source="api", error="未配置 api_key / base_url / model")

    quality = int(CONFIG.get("capture.jpeg_quality", 88))
    max_w = int(CONFIG.get("capture.max_width", 1280))
    content: List[Dict[str, Any]] = [{"type": "text", "text": USER_PROMPT}]
    for i, fr in enumerate(frames):
        jpg = encode_jpeg(fr, max_w, quality)
        if not jpg:
            continue
        content.append({"type": "text", "text": f"第 {i + 1} 帧："})
        content.append({"type": "image_url", "image_url": {"url": _to_data_url(jpg)}})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": cfg.get("max_tokens", 900),
    }

    # ---- DeepSeek 专属参数 ----
    # DeepSeek 的思考模式【默认开启】，会明显变慢、多烧 token，还会忽略 temperature。
    # 所以配置里 thinking: false 时要显式关掉。
    is_deepseek = "deepseek" in base.lower() or str(cfg.get("provider", "")).lower() == "deepseek"
    thinking = cfg.get("thinking", None)
    if is_deepseek and thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if not (is_deepseek and thinking):
        payload["temperature"] = cfg.get("temperature", 0.1)
    if cfg.get("json_mode", False):
        payload["response_format"] = {"type": "json_object"}

    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    last_err = ""
    for attempt in range(int(cfg.get("retries", 2)) + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.get("timeout", 60))
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                # 换服务商时，有的端点不认识 json_object / thinking 参数 → 摘掉再试一次
                if resp.status_code == 400:
                    low = resp.text.lower()
                    dropped = False
                    if "response_format" in low and "response_format" in payload:
                        payload.pop("response_format", None)
                        dropped = True
                    if "thinking" in low and "thinking" in payload:
                        payload.pop("thinking", None)
                        dropped = True
                    if dropped:
                        continue
                if resp.status_code in (400, 401, 403, 404):
                    break
            else:
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                if isinstance(text, list):  # 少数厂商返回分段结构
                    text = "".join(seg.get("text", "") for seg in text if isinstance(seg, dict))
                parsed = _parse_json(text)
                if parsed is None:
                    last_err = f"返回内容无法解析为JSON: {str(text)[:160]}"
                else:
                    return _build_api_result(parsed)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < int(cfg.get("retries", 2)):
            time.sleep(1.5 * (attempt + 1))

    return VisionResult(ok=False, source="api", error=last_err)


def _build_api_result(d: Dict[str, Any]) -> VisionResult:
    r = VisionResult(ok=True, source="api", raw=d)
    r.has_person = _bool(d.get("has_person"))
    r.face_visible = _bool(d.get("face_visible"))
    r.eyes_visible = _bool(d.get("eyes_visible"))
    try:
        r.eye_count = int(d.get("eye_count") or 0)
    except (TypeError, ValueError):
        r.eye_count = 0
    r.face_ratio = _num(d.get("face_ratio"))
    r.eye_area_quality = _num(d.get("eye_area_quality"))
    r.beauty_score = _num(d.get("beauty_score"))
    r.lens_suitable = _num(d.get("lens_suitable"))
    r.content_type = str(d.get("content_type") or "").strip()
    r.is_ad_or_live = _bool(d.get("is_ad_or_live"))
    r.theme = str(d.get("theme") or "").strip()
    r.lens_content = _bool(d.get("lens_content"))
    raw_hits = d.get("lens_brand_hits")
    if isinstance(raw_hits, str):
        raw_hits = re.split(r"[、,，;；/\s]+", raw_hits)
    if isinstance(raw_hits, (list, tuple)):
        r.lens_brand_hits = [str(x).strip() for x in raw_hits if str(x).strip()][:12]
    if r.lens_brand_hits:
        r.lens_content = True      # 列了词就说明确实是美瞳内容
    r.reason = str(d.get("reason") or "").strip()
    r.confidence = _num(d.get("confidence"), 0.6)
    r.occlusion = 0.0 if r.face_visible else 1.0

    # 一致性兜底：模型偶尔会自相矛盾，这里做一次修正
    if not r.has_person:
        r.lens_suitable = min(r.lens_suitable, 5.0)
    if not r.eyes_visible:
        r.lens_suitable = min(r.lens_suitable, 25.0)
    r.lens_suitable = max(0.0, min(100.0, r.lens_suitable))
    return r


# ------------------------------------------------------------------ 统一入口
def analyze(frames: List[np.ndarray], logger=None) -> VisionResult:
    """对一条视频的多帧做视觉评估。"""
    if not frames:
        return VisionResult(ok=False, source="none", error="没有可用画面")

    main_frame = frames[0]

    # 1) 本地预筛（省钱）
    pf = prefilter(main_frame)
    if not pf.passed:
        res = VisionResult(ok=True, source="prefilter", reason=f"本地预筛不通过：{pf.reason}",
                           face_ratio=round(pf.face_ratio, 4), lens_suitable=0.0, confidence=0.5)
        if logger:
            logger.debug(f"预筛拦截：{pf.reason}")
        return res

    # 2) 多模态 API
    if CONFIG.get("vision.enabled", True):
        res = api_analyze(frames)
        if res.ok:
            return res
        if logger:
            logger.warning(f"视觉API调用失败，尝试本地降级：{res.error}")
        if not CONFIG.get("vision.fallback_local", True):
            return res

    # 3) 本地降级
    return local_analyze(main_frame)
