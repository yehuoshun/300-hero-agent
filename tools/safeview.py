"""把截帧裁成「只看界面、不看人」的安全视图，专门给模型看。

为什么需要它：《300英雄》是二次元游戏，大厅主视觉、选英雄卡面、加载页/结算页原画、皮肤展示
都含角色立绘，其中部分可能不适合上传到外部模型。而本项目的自动化只需要**界面读数**
（阶段面板、比分、血量、灵魂数、小地图、技能栏），这些区域里没有立绘。

规矩（写进 AGENTS.md）：
- 原始整帧只落 `tmp/raw/`，**永远不 read_image**、不进对话、不入库；
- 只把 `tools/safeview.py` 生成到 `tmp/peek/` 的裁剪图给模型看；
- 万不得已要看全场，就用 `--mode wide`：整帧先按人像矩形打码（像素化）再降采样。

用法：
    python tools/safeview.py --in tmp/raw/f.png --mode phase
    python tools/safeview.py --in tmp/raw/f.png --mode all --out tmp/peek/f.png
    python tools/safeview.py --in tmp/raw/f.png --mode wide --out tmp/peek/f-wide.png

    --mode phase|score|minimap|hud|soul|chat   单块
    --mode all                                  上列全部拼成一张联络表
    --mode wide                                 整帧打码后缩到 1280 宽

坐标全部是相对客户区的比例，和对局 2560×1440 / 大厅 1280×800 都通用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 白名单：界面读数区，不含立绘
ZONES: dict[str, tuple[float, float, float, float]] = {
    # 左侧阶段面板：阶段名+倒计时+收集/提交数+祭灵者数值
    "phase": (0.000, 0.250, 0.170, 0.560),
    # 顶部比分与游戏时钟
    "score": (0.280, 0.000, 0.720, 0.060),
    # 右下小地图
    "minimap": (0.890, 0.810, 1.000, 1.000),
    # 底部自身血量/技能栏
    "hud": (0.280, 0.920, 0.720, 1.000),
    # 右侧灵魂属性两块面板
    "soul": (0.815, 0.370, 1.000, 0.570),
    # 左下聊天/系统提示（含玩家昵称，建议只自己看）
    "chat": (0.000, 0.560, 0.180, 0.720),
}

# 黑名单：整帧模式里必须打码的区域（立绘/皮肤/头像大图/活动广告图）
PORTRAIT_ZONES: tuple[tuple[float, float, float, float], ...] = (
    (0.000, 0.000, 1.000, 0.060),   # 顶部横幅/活动图
    (0.000, 0.045, 0.165, 0.205),   # 左上「战令/皮肤」小图
    (0.180, 0.080, 0.820, 0.870),   # 大厅主视觉大立绘
    (0.560, 0.060, 1.000, 0.560),   # 选英雄页右侧立绘/皮肤
    (0.745, 0.085, 1.000, 0.370),   # 右侧活动/皮肤广告图
    (0.000, 0.700, 0.240, 1.000),   # 左下头像卡
    (0.860, 0.700, 1.000, 1.000),   # 右下角头像/表情
)


def read_img(path: Path) -> np.ndarray:
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不出图：{path}")
    return img


def crop(img: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect
    a, b = int(x0 * w), int(y0 * h)
    c, d = int(x1 * w), int(y1 * h)
    return img[b:max(d, b + 1), a:max(c, a + 1)]


def pixelate(img: np.ndarray, blocks: int = 18) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, w // blocks), max(2, h // blocks)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def safe_wide(img: np.ndarray, width: int = 1280) -> np.ndarray:
    """整帧版：先把人像区像素化，再整体缩小。"""
    out = img.copy()
    for rect in PORTRAIT_ZONES:
        x0, y0, x1, y1 = rect
        h, w = out.shape[:2]
        out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = pixelate(
            out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], blocks=26)
    h, w = out.shape[:2]
    return cv2.resize(out, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def montage(img: np.ndarray, names: list[str], cell_w: int = 640) -> np.ndarray:
    tiles = []
    for name in names:
        t = crop(img, ZONES[name])
        h, w = t.shape[:2]
        scale = cell_w / w
        t = cv2.resize(t, (cell_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    width = cell_w * 2
    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        if len(pair) == 1:
            pad = np.zeros_like(pair[0])
            pad[:, :] = 40
            pair.append(pad)
        h = max(t.shape[0] for t in pair)
        pair = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                for t in pair]
        rows.append(np.hstack(pair))
    return np.vstack(rows)


def write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"写不出图：{path}")
    path.write_bytes(buf.tobytes())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="裁出「只看界面不看人」的安全视图")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--mode", default="all",
                    help="phase|score|minimap|hud|soul|chat|all|wide，逗号可多选")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    img = read_img(src)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if modes == ["wide"]:
        out = safe_wide(img)
        name = "wide"
    elif modes == ["all"]:
        out = montage(img, list(ZONES))
        name = "all"
    elif all(m in ZONES for m in modes):
        out = montage(img, modes)
        name = "+".join(modes)
    else:
        raise SystemExit(f"不认识的 mode：{args.mode}（可选 {', '.join(ZONES)}|all|wide）")

    dst = Path(args.dst) if args.dst else (ROOT / "tmp/peek" / f"{src.stem}-{name}.png")
    if not dst.is_absolute():
        dst = ROOT / dst
    write(dst, out)
    h, w = out.shape[:2]
    print(f"{dst}  {w}x{h}  mode={args.mode}  (整帧留在 {src.name}，不要上传)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
