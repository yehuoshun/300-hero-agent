"""启动游戏：窗口在就不动，窗口不在就按 config.json 拉起来。

只做三件事：找窗口、起进程、必要时自提权。不做更新、不点登录、不碰客户端文件。

两个实测坑（2026-11-13）：
1. 300.exe 清单是 requireAdministrator，非提权 Popen 直接 WinError 740；
   而且游戏进程跑在高完整性级别，非提权进程发的键鼠会被 UIPI 吞掉 —— 所以本进程也得提权。
2. 窗口标题会变：加载时是「300英雄」，之后可能变成「《300英雄》死火海阵营战开启…」这类活动标题。
   只按标题前缀匹配会找不到窗口（还会被「如何让AI玩300英雄 — DeepSeek Harness」这种标题骗到），
   所以主判据改成「窗口属于 300.exe 这个进程」，标题前缀只当兜底。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def fix_console() -> None:
    """对齐控制台编码：Windows 默认 GBK，Python 若按 UTF-8 输出会变乱码。

    只在真的写控制台时改：输出被重定向到日志文件时必须保持 UTF-8，
    否则提权子进程写出来的 logs/*.log 是一坨 GBK，读的人（和 AI）都看不懂。
    """
    if sys.platform != "win32":
        return
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return
    try:
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        enc = f"cp{cp}" if cp else "utf-8"
        sys.stdout.reconfigure(encoding=enc, errors="replace")
        sys.stderr.reconfigure(encoding=enc, errors="replace")
    except Exception:
        pass


def pid_image_name(pid: int) -> str:
    """进程可执行文件名（小写）。提权进程用 LIMITED 权限也能问到。"""
    if pid <= 0:
        return ""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name.lower()
        return ""
    finally:
        kernel32.CloseHandle(h)


def find_windows(title: str, process: str | None = None) -> list[tuple[int, str, int]]:
    """可见顶层窗口，返回 [(hwnd, 标题, pid)]。

    判定：窗口属于 process（如 300.exe）或 标题以 title 开头。
    """
    want = (process or "").lower()
    hits: list[tuple[int, str, int]] = []

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        text = buf.value.strip()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        by_title = bool(text) and text.startswith(title)
        by_proc = bool(want) and pid_image_name(pid.value) == want
        if by_title or by_proc:
            hits.append((int(hwnd), text, int(pid.value)))
        return True

    user32.EnumWindows(_EnumProc(_cb), 0)
    return hits


def load_config(path: Path = CONFIG_PATH) -> dict:
    # utf-8-sig：PowerShell 5.1 的 Set-Content -Encoding utf8 会写 BOM，json 会因此报
    # "Unexpected UTF-8 BOM"。手动改这个文件时很容易踩到，所以这里直接容忍 BOM。
    return json.loads(path.read_text(encoding="utf-8-sig"))


def window_spec(cfg: dict) -> tuple[str, str]:
    w = cfg["game"].get("window") or {}
    title = w.get("title") or cfg["game"].get("window_title") or "300英雄"
    process = w.get("process", "300.exe")
    return title, process


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def launch_command(cfg: dict) -> tuple[list[str], str]:
    """按 config 里选的启动方式拼命令，返回 (argv, cwd)"""
    la = cfg["game"]["launch"]
    name = la.get("method", "official")
    spec = la["methods"].get(name)
    if spec is None:
        raise SystemExit(f"config.json 没有启动方式 {name!r}，可选：{list(la['methods'])}")
    exe = Path(spec["exe"])
    if not exe.exists():
        raise SystemExit(f"启动器不存在：{exe}")
    cwd = spec.get("cwd") or str(exe.parent)
    return [str(exe), *spec.get("args", [])], cwd


def _shell_execute_runas(target: str, params: str, cwd: str) -> int:
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", target, params or None, cwd, 1)
    return int(rc)


def relaunch_elevated(script: Path, argv: list[str], log_path: Path) -> int:
    """把自己以管理员身份重启一份（UAC 只弹一次），当前进程退出。

    argv 只放业务参数，脚本路径单独传：早先漏了脚本路径，
    提权后变成 `python.exe --elevated ...`，Python 会把 --elevated 当成自己的选项。
    """
    params = subprocess.list2cmdline([str(script), *argv, "--elevated", "--log", str(log_path)])
    rc = _shell_execute_runas(sys.executable, params, str(ROOT))
    if rc <= 32:
        print(f"提权失败（ShellExecute 返回 {rc}，1223=用户点了取消）")
        return 2
    print(f"已请求提权，UAC 弹窗里点「是」。新进程输出写到 {log_path}")
    return 0


def strip_log_args(argv: list[str]) -> list[str]:
    """去掉 --log <path> 这一对，提权时会重新指定。"""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--log":
            skip = True
            continue
        out.append(a)
    return out


def ensure_running(cfg: dict | None = None, *, dry_run: bool = False) -> dict:
    """保证游戏窗口存在。返回 {hwnd, title, pid, launched, reason?}，hwnd=None 表示没等到。"""
    cfg = cfg or load_config()
    title, process = window_spec(cfg)
    la = cfg["game"]["launch"]

    hits = find_windows(title, process)
    if hits:
        hwnd, text, pid = hits[0]
        return {"launched": False, "hwnd": hwnd, "title": text, "pid": pid}

    if not la.get("enabled", True):
        return {"launched": False, "hwnd": None, "title": None, "pid": None,
                "reason": "launch.enabled 是 false"}

    argv, cwd = launch_command(cfg)
    if dry_run:
        return {"launched": False, "hwnd": None, "title": None, "pid": None,
                "would_run": argv, "cwd": cwd}

    print(f"窗口没找到，启动：{argv[0]}")
    try:
        subprocess.Popen(argv, cwd=cwd, close_fds=True)
    except OSError as e:
        if getattr(e, "winerror", None) != 740:
            raise
        print("  这个程序要求管理员权限，正在弹 UAC —— 请在弹窗里点「是」")
        rc = _shell_execute_runas(argv[0], subprocess.list2cmdline(argv[1:]), cwd)
        if rc <= 32:
            return {"launched": False, "hwnd": None, "title": None, "pid": None,
                    "reason": f"提权启动失败，ShellExecute 返回 {rc}"}

    deadline = time.time() + float(la.get("wait_seconds", 150))
    poll = float(la.get("poll_seconds", 1.0))
    while time.time() < deadline:
        hits = find_windows(title, process)
        if hits:
            hwnd, text, pid = hits[0]
            print(f"窗口出现：{text!r}  hwnd={hwnd} pid={pid}")
            return {"launched": True, "hwnd": hwnd, "title": text, "pid": pid}
        time.sleep(poll)
    return {"launched": True, "hwnd": None, "title": None, "pid": None,
            "reason": f"等了 {int(la.get('wait_seconds', 150))} 秒没等到窗口"}


def main(argv: list[str]) -> int:
    log_path = None
    if "--log" in argv:
        i = argv.index("--log")
        if i + 1 < len(argv):
            log_path = Path(argv[i + 1])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_path, "a", encoding="utf-8", buffering=1)
            sys.stdout = f
            sys.stderr = f
    fix_console()
    cfg = load_config()
    title, process = window_spec(cfg)

    if "--check" in argv:
        hits = find_windows(title, process)
        if hits:
            for hwnd, text, pid in hits:
                print(f"找到窗口：{text!r}  hwnd={hwnd} pid={pid}")
            return 0
        print(f"没找到：进程 {process} 的窗口，也没有标题以「{title}」开头的可见窗口")
        return 1

    if "--dry-run" in argv:
        r = ensure_running(cfg, dry_run=True)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["hwnd"] else 1

    if not is_admin() and "--elevated" not in argv and cfg["game"]["launch"].get("elevate", True):
        return relaunch_elevated(Path(__file__).resolve(), strip_log_args(argv),
                                 log_path or (ROOT / "logs" / "launch.log"))

    r = ensure_running(cfg)
    print(f"是否管理员：{is_admin()}")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r["hwnd"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
