"""把截帧裁成「该看的看、不该看的遮」的视图，供模型读画面。

为什么要分层（2026-11-19 用户定调）：
- **对局内必须看全画面** —— 不看画面就没法跟人博弈（走位、放技能、判断团战），所以对局帧**不打码**；
- **只有含角色立绘/卡面的页面才需要遮**：大厅首页大立绘、选英雄页的英雄卡网格与皮肤预览、活动/广告图；
  而选英雄本身只用「随机」或搜索框，不需要看清卡面；
- 原始整帧仍只落 `tmp/raw/`，给模型看的是 `tmp/peek/` 下由本工具产出的图。

用法：
    python tools/safeview.py --in tmp/raw/f.png --mode game            # 对局：整流缩小，不打码
    python tools/safeview.py --in tmp/raw/p.png --mode pick            # 选英雄页：只留顶部倒计时与底部按钮
    python tools/safeview.py --in tmp/raw/h.png --mode wide            # 大厅/菜单：立绘区像素化
    python tools/safeview.py --in tmp/raw/f.png --mode phase,minimap    # 对局：只裁读数区
    python tools/safeview.py --in tmp/raw/f.png --mode all             # 对局：读数区拼成联络表

    --mode phase|score|minimap|hud|soul|chat   单块读数区（无立绘）
    --mode all                                  上列全部拼成一张
    --mode game                                 对局整帧（缩到 1600 宽，不打码）
    --mode pick                                 选英雄页：顶部倒计时 + 底部「出击」
    --mode wide                                 整帧打码（立绘区像素化）后缩到 1280 宽

坐标全是相对客户区的比例，对局 2560×1440 / 大厅 1280×800 通用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 白名单：界面读数区，不含立绘（对局内）
ZONES: dict[str, tuple[float, float, float, float]] = {
    "phase": (0.000, 0.250, 0.170, 0.560),    # 左侧阶段面板
    "score": (0.280, 0.000, 0.720, 0.060),    # 顶部比分/时钟
    "minimap": (0.890, 0.810, 1.000, 1.000),  # 右下小地图
    "hud": (0.280, 0.920, 0.720, 1.000),      # 底部血量/技能
    "soul": (0.815, 0.370, 1.000, 0.570),     # 右侧灵魂属性
    "chat": (0.000, 0.560, 0.180, 0.720),     # 左下聊天（含昵称，慎传）
}

# 选英雄页只保留「可以操作」的两条带：顶部等级/倒计时、底部出击按钮
PICK_ZONES: dict[str, tuple[float, float, float, float]] = {
    "pick-top": (0.300, 0.000, 0.760, 0.095),
    "pick-bottom": (0.000, 0.880, 1.000, 1.000),
}

# 黑名单：大厅/菜单整帧模式下必须打码的区域（立绘、卡面、广告图）
PORTRAIT_ZONES: tuple[tuple[float, float, float, float], ...] = (
    (0.000, 0.000, 1.000, 0.060),   # 顶部横幅/活动图
    (0.000, 0.045, 0.165, 0.205),   # 左上战令/皮肤小图
    (0.180, 0.080, 0.820, 0.870),   # 大厅主视觉大立绘
    (0.040, 0.270, 0.530, 0.690),   # 选英雄页英雄卡网格
    (0.560, 0.060, 1.000, 0.560),   # 选英雄页右侧立绘/皮肤
    (0.590, 0.590, 0.880, 0.870),   # 选英雄页皮肤预览框
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


def shrink(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def safe_wide(img: np.ndarray, width: int = 1280) -> np.ndarray:
    """整帧版：先把立绘/卡面区像素化，再整体缩小。"""
    out = img.copy()
    h, w = out.shape[:2]
    for rect in PORTRAIT_ZONES:
        x0, y0, x1, y1 = rect
        a, b, c, d = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        if c <= a or d <= b:
            continue
        out[b:d, a:c] = pixelate(out[b:d, a:c], blocks=26)
    return shrink(out, width)


def montage(img: np.ndarray, parts: list[tuple[str, tuple[float, float, float, float]]],
            cell_w: int = 640) -> np.ndarray:
    tiles = []
    for name, rect in parts:
        t = crop(img, rect)
        h, w = t.shape[:2]
        scale = cell_w / w
        t = cv2.resize(t, (cell_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        h = max(t.shape[0] for t in pair)
        pair = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                   value=(40, 40, 40)) for t in pair]
        if len(pair) == 1:
            pad = np.full_like(pair[0], 40)
            pair.append(pad)
        rows.append(np.hstack(pair))
    return np.vstack(rows)


def write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"写不出图：{path}")
    path.write_bytes(buf.tobytes())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="裁出「该看的看、该遮的遮」的画面视图")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--mode", default="game",
                    help="game|pick|wide|phase|score|minimap|hud|soul|chat|all，逗号可多选")
    ap.add_argument("--width", type=int, default=1600, help="game/wide 的输出宽度")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    img = read_img(src)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if modes == ["game"]:
        out, name = shrink(img, args.width), "game"
    elif modes == ["wide"]:
        out, name = safe_wide(img, min(args.width, 1280)), "wide"
    elif modes == ["pick"]:
        out = montage(img, list(PICK_ZONES.items()))
        name = "pick"
    elif modes == ["all"]:
        out = montage(img, list(ZONES.items()))
        name = "all"
    elif all(m in ZONES for m in modes):
        out = montage(img, [(m, ZONES[m]) for m in modes])
        name = "+".join(modes)
    else:
        raise SystemExit(f"不认识的 mode：{args.mode}（可选 game/pick/wide/{', '.join(ZONES)}/all）")

    dst = Path(args.dst) if args.dst else (ROOT / "tmp/peek" / f"{src.stem}-{name}.png")
    if not dst.is_absolute():
        dst = ROOT / dst
    write(dst, out)
    h, w = out.shape[:2]
    print(f"{dst}  {w}x{h}  mode={args.mode}  (整帧留在 {src.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
