"""浠庝竴寮犲抚閲岃涓€鍧楁斁澶у瓨鐩橈紝鏂逛究鐢ㄧ溂鐫涢噺 HUD 鐨勫儚绱犱綅缃€?
    python tools/crop.py tmp/frame.png tmp/crop.png 0 1150 700 290 --scale 2

鍙傛暟锛?杈撳叆鍥? <杈撳嚭鍥? <left> <top> <width> <height> [--scale N]
鍧愭爣鏄鍥捐嚜韬殑鍍忕礌鍧愭爣锛堜笉鍋氱浉瀵规瘮渚嬫崲绠楋紝閲?HUD 鐢ㄤ笉鐫€锛夈€?"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str]) -> int:
    if len(argv) < 6:
        print(__doc__)
        return 0
    src, dst = Path(argv[0]), Path(argv[1])
    left, top, w, h = (int(v) for v in argv[2:6])
    scale = 1.0
    if "--scale" in argv:
        scale = float(argv[argv.index("--scale") + 1])
    # cv2.imread 涓嶈闈?ASCII 璺緞锛堟湰椤圭洰鍦ㄥ伐浣滃尯涓枃鐩綍涓嬶級锛岃嚜宸辫瀛楄妭鍐嶈В
    img = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"璇讳笉鍒?{src}")
        return 1
    H, W = img.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(W, left + w), min(H, top + h)
    crop = img[top:bottom, left:right]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(dst.suffix or ".png", crop)
    if not ok:
        print("缂栫爜澶辫触")
        return 1
    dst.write_bytes(buf.tobytes())
    print(f"{src.name} 鐨?({left},{top})-({right},{bottom}) 鈫?{dst}  {crop.shape[1]}x{crop.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

