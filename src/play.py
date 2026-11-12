"""最小闭环：抓帧 → 找敌方红血条 → 右键过去打 / 放技能；F12 急停。

只用眼睛和手：不读内存、不注入 DLL、不改客户端。
这一步的策略还非常粗糙，目标是先让「看→动」这条线跑通：
- 视野里有红血条 → 右键点最近的那条（300英雄 右键＝移动/攻击）
- 视野里没有 → 右键点屏幕右侧一带，往敌方方向推
- 贴脸时按 w / q（没命中、没冷却也不亏）

已知不够用的地方（下次改）：自己血量读不出、不知道敌我在哪、不看小地图。
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from capture import _imwrite_unicode, focus, grab
from input import click_client, press_key
from launch import (ROOT, find_windows, fix_console, is_admin, load_config,
                    relaunch_elevated, strip_log_args, window_spec)

user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_F12 = 0x7B


def emergency_stopped() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F12) & 0x8000)


def find_red_bars(frame: np.ndarray) -> list[dict]:
    """找敌方头顶血条：细长、高饱和的红条。坐标是整帧像素。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    mask = (((hh <= 10) | (hh >= 170)) & (ss >= 110) & (vv >= 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if bw < w * 0.015 or bw > w * 0.25:
            continue
        if bh < 2 or bh > h * 0.03:
            continue
        if bw / max(bh, 1) < 4:
            continue
        if area < 0.5 * bw * bh:
            continue
        cx, cy = (float(v) for v in cents[i])
        if cy < h * 0.10:                       # 顶栏计分板
            continue
        if cx > w * 0.88 and cy > h * 0.72:     # 右下小地图
            continue
        if cy > h * 0.92:                       # 底部技能栏
            continue
        out.append({"x": x, "y": y, "w": bw, "h": bh, "cx": cx, "cy": cy})
    return out


def main(argv: list[str]) -> int:
    if "--log" in argv:
        i = argv.index("--log")
        log_path = Path(argv[i + 1])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f

    fix_console()
    cfg = load_config()
    title, process = window_spec(cfg)

    duration = 120.0
    if "--duration" in argv:
        duration = float(argv[argv.index("--duration") + 1])
    shot = None
    if "--shot" in argv:
        shot = argv[argv.index("--shot") + 1]

    hits = find_windows(title, process)
    if not hits:
        print(f"没找到游戏窗口（进程 {process}）")
        return 1
    hwnd, text, pid = hits[0]

    if not is_admin() and "--elevated" not in argv:
        return relaunch_elevated(__file__, strip_log_args(argv), ROOT / "logs" / "play.log")

    print(f"窗口：{text!r} hwnd={hwnd} pid={pid} 管理员={is_admin()}")
    if not focus(hwnd):
        print("警告：没抢到前台，键鼠可能发不进去")

    n = 0
    t0 = time.time()
    while time.time() - t0 < duration:
        if emergency_stopped():
            print(f"F12 急停，已跑 {n} 拍")
            break
        frame, _box = grab(hwnd)
        fh, fw = frame.shape[:2]
        bars = find_red_bars(frame)
        bars.sort(key=lambda b: (b["cx"] - fw / 2) ** 2 + (b["cy"] - fh / 2) ** 2)
        if bars:
            t = bars[0]
            click_client(hwnd, t["cx"] / fw, (t["cy"] + 10) / fh, button="right")
            dist = abs(t["cx"] - fw / 2)
            if dist < fw * 0.18:
                press_key("w")
                press_key("q")
            print(f"[{n:03d}] 打 {t['w']}x{t['h']} @({t['cx']:.0f},{t['cy']:.0f}) 距中 {dist:.0f}px")
        else:
            click_client(hwnd, 0.62, 0.52, button="right")
            print(f"[{n:03d}] 没看到红条，往右推")
        n += 1
        time.sleep(0.6)

    if shot:
        img, _ = grab(hwnd)
        if _imwrite_unicode(ROOT / shot, img):
            print(f"结束截图：{shot}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
