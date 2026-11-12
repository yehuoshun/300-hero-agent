"""抓客户区：窗口 → 客户区屏幕矩形 → mss 抓一帧 → BGR ndarray（H,W,3）。

坐标原点约定＝客户区左上角，配置里只存相对比例；这里给出的 box 只用于喂 mss。
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

import mss
import numpy as np

from launch import ROOT, find_windows, fix_console, load_config, window_spec

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SW_RESTORE = 9


def focus(hwnd: int, wait: float = 0.3, tries: int = 3) -> bool:
    """把窗口提到前台（mss 抓的是屏幕，被别的窗口盖住就抓到别人）。

    SetForegroundWindow 有「前台锁」：不是当前前台进程调它只会闪任务栏。
    标准绕法是把自己 attach 到前台窗口的输入队列再抢。返回是否真的抢到了。
    用户一直在别的软件里点鼠标时，抢到也会被立刻抢回去，所以重试几次。
    """
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.2)
    for attempt in range(tries):
        if user32.GetForegroundWindow() == hwnd:
            return True
        fg = user32.GetForegroundWindow()
        tid_fg = user32.GetWindowThreadProcessId(fg, None)
        tid_me = kernel32.GetCurrentThreadId()
        attached = bool(user32.AttachThreadInput(tid_me, tid_fg, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(tid_me, tid_fg, False)
        time.sleep(wait if attempt == tries - 1 else 0.15)
    return user32.GetForegroundWindow() == hwnd



def enable_dpi_awareness() -> None:
    """不做 DPI 虚拟化，否则 150% 缩放下拿到的坐标是假的。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def client_rect(hwnd: int) -> dict:
    """客户区在屏幕上的矩形（物理像素）。"""
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        raise OSError(ctypes.get_last_error(), "GetClientRect 失败")
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise OSError(ctypes.get_last_error(), "ClientToScreen 失败")
    return {"left": origin.x, "top": origin.y,
            "width": rc.right - rc.left, "height": rc.bottom - rc.top}


def grab(hwnd: int) -> tuple[np.ndarray, dict]:
    """抓一帧，返回 (BGR 图, 客户区 box)。"""
    box = client_rect(hwnd)
    if box["width"] <= 0 or box["height"] <= 0:
        raise RuntimeError(f"客户区尺寸异常：{box}（窗口可能最小化了）")
    with mss.MSS() as sct:
        raw = sct.grab(box)
    return np.asarray(raw)[:, :, :3].copy(), box


def find_game_window() -> tuple[int, str]:
    cfg = load_config()
    title, process = window_spec(cfg)
    hits = find_windows(title, process)
    if not hits:
        raise SystemExit(f"没找到游戏窗口（进程 {process} / 标题前缀「{title}」），先跑 src/launch.py")
    hwnd, text, _pid = hits[0]
    return hwnd, text


def main(argv: list[str]) -> int:
    fix_console()
    enable_dpi_awareness()
    out = ROOT / "tmp" / "client.png"
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])

    hwnd, text = find_game_window()
    if "--no-focus" not in argv:
        if focus(hwnd):
            print("已把游戏窗口提到前台")
        else:
            print("警告：没抢到前台，这一帧可能是别的窗口（mss 抓的是屏幕，不是窗口内容）")
    img, box = grab(hwnd)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not _imwrite_unicode(out, img):
        raise SystemExit(f"写文件失败：{out}")
    print(f"窗口：{text!r}  hwnd={hwnd}")
    print(f"客户区：{box}")
    print(f"帧：{img.shape[1]}x{img.shape[0]}  dtype={img.dtype}")
    print(f"存到：{out}")
    return 0


def _imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """cv2.imwrite 不认非 ASCII 路径，自己编码写。"""
    try:
        import cv2
    except ImportError:
        from PIL import Image
        Image.fromarray(img[:, :, ::-1]).save(str(path))
        return True
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
