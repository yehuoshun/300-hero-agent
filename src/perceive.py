"""看：从一帧里读「自己」的信息。

这里只放**能量化、能自证**的东西。每个几何常量都标了它是从哪张图、哪个像素量出来的，
换分辨率或游戏改版后必须重量。

已确定：
- 自身血量＝屏幕底部中央那条**绿色**进度条（对局截图里带 `9252/9252` 数字）。
  轨道左端固定在客户区 x≈0.4047，满血右端 x≈0.4980，y≈0.99。
  这条不是"宽绿条贴着左下角"——左下角那条紫红色的是**经验条**（满级 18 级时它是空的，
  死亡时也还在涨，所以它不可能是血条；见 2026-11-13 量测）。

未确定（别当已知用）：
- 自身位置。这类俯视场均锁定镜头把英雄放屏幕正中，本模块先按 (0.5, 0.5) 返回，
  **尚未实测**。等能在训练模式里造场景了再验。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# 血条轨道：量自 2026-10-29_21-3-16.bmp（2560x1440）绝对像素 x 1036..1275, y 1426..1437
HP_TRACK = (0.4047, 0.9890, 0.4980, 0.9986)   # x0, y0, x1, y1，相对客户区
HP_GREEN_LO = (45, 110, 110)                  # H 45~85：从截图里取的绿条色相范围
HP_GREEN_HI = (85, 255, 255)


def _track_rect(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = HP_TRACK
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def self_hp(frame: np.ndarray) -> dict:
    """读自身血量比例。

    返回 {"ratio": 0~1 或 None, "rect": (x0,y0,x1,y1) 整帧像素, "note": 说明}。
    ratio 为 None 表示轨道里没有绿条——通常是自己已死（或者这张图不是对局画面）。
    判据是"列"而不是"像素总数"：进度条填充从轨道左端长出来，所以按列看更稳。
    """
    ax0, ay0, ax1, ay1 = _track_rect(frame)
    crop = frame[ay0:ay1, ax0:ax1]
    if crop.size == 0:
        return {"ratio": None, "rect": (ax0, ay0, ax1, ay1), "note": "轨道不在画面内"}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HP_GREEN_LO, HP_GREEN_HI) > 0
    total = crop.shape[1]
    if not mask.any():
        return {"ratio": None, "rect": (ax0, ay0, ax1, ay1), "note": "没有绿条（血量 0 或已死）"}
    # 血条上压着 `9252/9252` 白字，会把某几行的绿条切断；所以按「列」看：
    # 一列里只要还有一部分是绿的，就算这一列被填了。
    filled = mask.mean(axis=0) > 0.20
    idx = np.flatnonzero(filled)
    if idx.size == 0:
        return {"ratio": None, "rect": (ax0, ay0, ax1, ay1), "note": "没有绿条（血量 0 或已死）"}
    start, end = int(idx[0]), int(idx[-1])
    # 填充必须贴着轨道左端。HUD 是按浮点坐标画的，实测不同帧左边缘会差 1~2 px，
    # 所以允许 4 px 或 3% 以内的偏移，超过就当「抠错了地方」，宁可报 None 也不报假血。
    if start > max(4, int(total * 0.03)):
        return {"ratio": None, "rect": (ax0, ay0, ax1, ay1),
                "note": f"绿条不在轨道左端（偏了 {start}px），位置可疑"}
    ratio = min(1.0, (end + 1 - start) / (total - start))
    return {"ratio": ratio, "rect": (ax0, ay0, ax1, ay1), "note": ""}


def self_position(frame: np.ndarray) -> tuple[float, float]:
    """自身在屏幕上的相对位置。锁定镜头的俯视游戏＝屏幕正中，**尚未实测**。"""
    return 0.5, 0.5


def _debug(img_path: str, out_path: str | None) -> int:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"读不到 {img_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    info = self_hp(img)
    x0, y0, x1, y1 = info["rect"]
    h, w = img.shape[:2]
    up = 6
    crop = img[max(0, y0 - 6):min(h, y1 + 6), max(0, x0 - 10):min(w, x1 + 10)]
    big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(big, (10 * up, 6 * up), ((10 + x1 - x0) * up, (6 + y1 - y0) * up), (0, 255, 0), 2)
    print(f"轨道 {x1-x0}x{y1-y0} @({x0},{y0})  血量={info['ratio']}  {info['note']}")
    cv2.putText(big, f"hp={info['ratio']}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
    out = Path(out_path) if out_path else Path(__file__).resolve().parent.parent / "tmp" / "hp_debug.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), big)
    print(f"写了 {out}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--debug" in args:
        i = args.index("--debug")
        out = args[args.index("--out") + 1] if "--out" in args else None
        sys.exit(_debug(args[i + 1], out))
    print(__doc__)
