"""最小闭环：抓帧 → 读自己血量 → 找敌方红血条 → 右键过去打 / 放技能；F12 急停。

只用眼睛和手：不读内存、不注入 DLL、不改客户端。
这一步的策略还非常粗糙，目标是先让「看→动」这条线跑通：
- 每拍先读自身血量（底部中央绿条，见 perceive.self_hp），写进日志
- 视野里有红血条 → 右键点最近的那条（300英雄 右键＝移动/攻击）
- 贴脸且血够 → 按 w / q
- 读不到血量（＝已死或不在对局）→ 这一拍不动，只记录
- 视野里没有敌 → 右键点屏幕右侧一带，往敌方方向推

已知不够用的地方（下次改）：不知道敌我在哪、不看小地图、血少了还不会往回跑。
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
from perceive import self_hp, self_position

user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_F12 = 0x7B
LOW_HP = 0.25          # 低于这个比例就不再贴脸（策略里唯一用到血量的地方）


def emergency_stopped() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F12) & 0x8000)


def find_red_bars(frame: np.ndarray) -> list[dict]:
    """找敌方头顶血条：细长、高饱和的红条。坐标是整帧像素。

    剔掉 HUD 里的「假红条」：2026-11-19 对局里实测到，左侧「阶段面板」里也有红条状色块，
    老版本没剔它，对局循环于是整场对着屏幕左缘 (208,731) 点了 38 拍。
    HUD 固定占位（相对客户区，量自 2560×1440 帧）：
      左侧阶段面板 x<0.155w 且 y 0.25~0.56h；右侧灵魂属性双面板 x>0.82w 且 y 0.38~0.56h；
      顶部比分 y<0.10h；底部道具栏/技能栏 y>0.86h；右下小地图 x>0.88w 且 y>0.72h。
    按矩形剔、不砍整列：语料 61 张对局帧对比过，中位数 1 / 最大 8 与老版本一致，只有 7 帧数量变化。
    """
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
        # 左侧「阶段面板」：x<0.155w、y 0.25~0.56h，里面也有红条状色块
        if cx < w * 0.155 and h * 0.25 < cy < h * 0.56:
            continue
        # 右侧「灵魂属性」双面板：x>0.82w、y 0.38~0.56h
        if cx > w * 0.82 and h * 0.38 < cy < h * 0.56:
            continue
        if cy < h * 0.10:                       # 顶部比分/倒计时
            continue
        if cy > h * 0.86:                       # 底部道具栏与技能栏
            continue
        if cx > w * 0.88 and cy > h * 0.72:     # 右下小地图
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
        hp = self_hp(frame)
        px, py = self_position(frame)
        hp_txt = "?" if hp["ratio"] is None else f"{hp['ratio']:.2f}"
        # 已死（读不到血量）时不做决定，只记录；复活后血量会自己回来
        dead = hp["ratio"] is None
        bars = find_red_bars(frame)
        bars.sort(key=lambda b: (b["cx"] - fw / 2) ** 2 + (b["cy"] - fh / 2) ** 2)
        head = f"[{n:03d}] t={time.time() - t0:5.1f}s hp={hp_txt} 我@({px:.2f},{py:.2f}) 红条{len(bars)}"
        if dead:
            print(head + " 读不到血量，可能已死，本拍不动")
        elif bars:
            t = bars[0]
            click_client(hwnd, t["cx"] / fw, (t["cy"] + 10) / fh, button="right")
            dist = abs(t["cx"] - fw / 2)
            face = ""
            if dist < fw * 0.18:
                if hp["ratio"] < LOW_HP:
                    face = " 血少不贴脸"
                else:
                    press_key("w")
                    press_key("q")
                    face = " 贴脸放 wq"
            print(head + f" 打 {t['w']}x{t['h']} @({t['cx']:.0f},{t['cy']:.0f}) 距中 {dist:.0f}px" + face)
        else:
            click_client(hwnd, 0.62, 0.52, button="right")
            print(head + " 没看到红条，往右推")
        n += 1
        time.sleep(0.6)

    if shot:
        img, _ = grab(hwnd)
        if _imwrite_unicode(ROOT / shot, img):
            print(f"结束截图：{shot}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
