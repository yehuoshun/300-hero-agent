"""发键鼠：只走 SendInput，坐标一律「客户区相对比例」，不存绝对像素。

为什么要提权：300.exe 跑在高完整性级别，非提权进程的 SendInput 会被 UIPI 直接丢掉
（不抛错、就是没反应），所以这里发现自己不是管理员就自提权，UAC 只弹一次。
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

from capture import _imwrite_unicode, client_rect, enable_dpi_awareness, focus, grab
from launch import (ROOT, find_windows, fix_console, is_admin, load_config,
                    relaunch_elevated, strip_log_args, window_spec)

user32 = ctypes.WinDLL("user32", use_last_error=True)

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send_mouse(flags: int) -> None:
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi = MOUSEINPUT(0, 0, 0, flags, 0, 0)
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(ctypes.get_last_error(), f"SendInput 失败（flags={flags:#x}）")


def click_client(hwnd: int, xr: float, yr: float, *, hold: float = 0.05) -> dict:
    """在客户区相对位置 (xr, yr) 点一下左键。返回实际落点，便于核对。"""
    box = client_rect(hwnd)
    x = box["left"] + int(round(xr * box["width"]))
    y = box["top"] + int(round(yr * box["height"]))
    if not user32.SetCursorPos(x, y):
        raise OSError(ctypes.get_last_error(), "SetCursorPos 失败")
    time.sleep(0.04)
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(hold)
    _send_mouse(MOUSEEVENTF_LEFTUP)
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return {"target": (x, y), "cursor": (pt.x, pt.y), "box": box}


def main(argv: list[str]) -> int:
    log_path = None
    if "--log" in argv:
        i = argv.index("--log")
        log_path = Path(argv[i + 1])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f

    fix_console()
    enable_dpi_awareness()
    cfg = load_config()
    title, process = window_spec(cfg)

    hits = find_windows(title, process)
    if not hits:
        print(f"没找到游戏窗口（进程 {process}），先跑 src/launch.py")
        return 1
    hwnd, text, pid = hits[0]

    if not is_admin() and "--elevated" not in argv:
        return relaunch_elevated(Path(__file__).resolve(), strip_log_args(argv),
                                 ROOT / "logs" / "input.log")

    print(f"窗口：{text!r}  hwnd={hwnd} pid={pid}")
    print(f"是否管理员：{is_admin()}")
    if not focus(hwnd):
        print("警告：没抢到前台，点击可能落到别的窗口上")

    name = argv[argv.index("--click") + 1] if "--click" in argv else None
    if name:
        point = cfg["ui"].get(name)
        if point is None:
            print(f"config.json 的 ui 里没有 {name!r}")
            return 2
        r = click_client(hwnd, point[0], point[1])
        print(f"点击 {name} 相对={point} → 屏幕{(r['target'])} 光标{(r['cursor'])}")

    settle = float(cfg.get("ui", {}).get("click_settle", 4.0))
    if "--shot" in argv or name:
        time.sleep(settle)
        img, box = grab(hwnd)
        out = Path(argv[argv.index("--shot") + 1]) if "--shot" in argv else ROOT / "tmp" / "after-click.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not _imwrite_unicode(out, img):
            print(f"写文件失败：{out}")
            return 1
        print(f"点击后截图：{out}  {img.shape[1]}x{img.shape[0]}  客户区={box}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
