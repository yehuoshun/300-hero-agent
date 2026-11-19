"""把一堆帧拼成一张联络表（contact sheet），一次看 12 帧，省得一张张翻。

为什么需要：看单帧很贵，而「这 60 秒里场上发生了什么」往往只需要一眼扫过。
    python tools/contact.py out.png --cols 4 --cell 480 tmp/explore/obs-*.png

可选裁剪：--crop x,y,w,h（像素，整帧坐标系）。例如只盯左侧阶段面板：
    python tools/contact.py panel.png --crop 0,290,320,220 --cols 6 --cell 320 tmp/explore/panel-*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def read_img(path: Path) -> np.ndarray:
    """cv2.imread 在中文路径下返回 None，必须走 imdecode。"""
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到图 {path}")
    return img


def find_frames(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if any(ch in it for ch in "*?["):
            out.extend(sorted(p.parent.glob(p.name)))
        else:
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=480, help="每格宽（高按比例算）")
    ap.add_argument("--crop", default="", help="x,y,w,h 整帧像素；空＝整帧")
    ap.add_argument("--label", action="store_true", help="在每格左上角写文件名")
    a = ap.parse_args(argv)

    paths = find_frames(a.frames)
    if not paths:
        raise SystemExit("没找到帧")

    tiles = []
    for p in paths:
        img = read_img(p)
        if a.crop:
            x, y, w, h = (int(v) for v in a.crop.split(","))
            img = img[y:y + h, x:x + w]
        scale = a.cell / img.shape[1]
        tile = cv2.resize(img, (a.cell, max(1, int(img.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        if a.label:
            cv2.putText(tile, p.stem[-24:], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    rows = []
    for i in range(0, len(tiles), a.cols):
        row = tiles[i:i + a.cols]
        h = max(t.shape[0] for t in row)
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                  value=(40, 40, 40)) for t in row]
        while len(row) < a.cols:
            row.append(np.full_like(row[0], 40))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(out.suffix or ".png", sheet)
    if not ok:
        raise SystemExit("编码失败")
    out.write_bytes(buf.tobytes())
    print(f"{len(tiles)} 帧 → {out} {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
