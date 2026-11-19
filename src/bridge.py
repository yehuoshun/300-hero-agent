"""常驻提权桥：UAC 只弹一次，之后所有键鼠/截图/对局循环都从这个进程里发。

为什么需要它：300.exe 跑在高完整性级别，非提权进程发的 SendInput 会被 UIPI **静默吞掉**；
而每条命令都自提权＝每次都弹 UAC，既慢，又把用户的注意力反复扯断。
所以：起一个常驻的提权服务进程，非提权侧用 tools/ctl.py 发一行 JSON、收一行 JSON。

协议：127.0.0.1:<port>（默认 52341），一行一个 JSON 请求，回一行 JSON：

  {"cmd":"ping"}                              存活探测
  {"cmd":"state"}                             窗口/血量/红条数/对局循环是否在跑
  {"cmd":"click","ui":"hall_attack"}           按 config.ui 里的键点（左键）
  {"cmd":"click","point":[0.5,0.5],"button":"right"}   按客户区相对比例点
  {"cmd":"key","name":"w"}                    按一下键
  {"cmd":"wait","seconds":1}                  干等
  {"cmd":"shot","path":"tmp/a.png"}           抢前台→抓一帧→存盘
  {"cmd":"play","duration":120,"log":"tmp/play.log"}  起对局循环（后台线程）
  {"cmd":"stop"}                              让对局循环收手
  {"cmd":"quit"}                              退出桥

约定不变：坐标只走相对比例；抓图走 mss 抓客户区；F12 急停在循环里轮询。

启动（自己会提权，UAC 只这一次）：
    .venv\\Scripts\\python.exe src\\bridge.py --serve --hidden
"""

from __future__ import annotations

import ctypes
import json
import socket
import sys
import threading
import time
from pathlib import Path

import cv2

from capture import _imwrite_unicode, enable_dpi_awareness, focus, grab
from input import click_client, move_rel, press_key
from launch import (ROOT, find_windows, fix_console, is_admin, load_config,
                    window_spec)
from perceive import self_hp, self_position
from play import emergency_stopped, find_red_bars

HOST = "127.0.0.1"
PORT = 52341
LOW_HP = 0.25


class Game:
    """所有对游戏的动作都经过这里，保证「找窗口」这一件事只有一份实现。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.play_thread: threading.Thread | None = None
        self.last_log: list[str] = []

    # ---------- 基础 ----------
    def hwnd(self) -> tuple[int, str, dict]:
        cfg = load_config()
        title, process = window_spec(cfg)
        hits = find_windows(title, process)
        if not hits:
            raise RuntimeError(f"没找到游戏窗口（进程 {process}）")
        hwnd, text, _pid = hits[0]
        return hwnd, text, cfg

    def frame(self) -> tuple["object", dict]:
        hwnd, _text, _cfg = self.hwnd()
        ok = focus(hwnd)
        img, box = grab(hwnd)
        return img, {"focus": ok, "box": box}

    def shot(self, path: str) -> dict:
        img, info = self.frame()
        out = Path(path)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = _imwrite_unicode(out, img)
        return {"path": str(out), "size": [int(img.shape[1]), int(img.shape[0])],
                "focus": info["focus"], "box": info["box"], "ok": ok}

    def state(self) -> dict:
        img, info = self.frame()
        hp = self_hp(img)
        bars = find_red_bars(img)
        px, py = self_position(img)
        return {"hwnd": None, "focus": info["focus"], "box": info["box"],
                "size": [int(img.shape[1]), int(img.shape[0])],
                "hp": hp["ratio"], "hp_note": hp["note"],
                "me": [px, py], "red_bars": len(bars),
                "nearest": min((b["cx"] for b in bars), default=None),
                "playing": bool(self.play_thread and self.play_thread.is_alive()),
                "log_tail": self.last_log[-6:]}

    # ---------- 动作 ----------
    def click(self, ui: str | None, point, button: str, mode: str = "ui",
              hold: float = 0.06) -> dict:
        hwnd, _text, cfg = self.hwnd()
        if ui:
            point = cfg["ui"].get(ui)
            if point is None:
                raise RuntimeError(f"config.json 的 ui 里没有 {ui!r}")
        focus(hwnd)
        r = click_client(hwnd, float(point[0]), float(point[1]), button=button,
                         mode=mode, hold=float(hold))
        return {"clicked": ui or "point", "rel": list(point), "screen": r.get("target") or r.get("target_client"),
                "cursor": r["cursor"], "mode": mode}

    # ---------- 对局循环 ----------
    def start_play(self, duration: float, log: str | None, dry_run: bool = False) -> dict:
        if self.play_thread and self.play_thread.is_alive():
            return {"playing": True, "note": "已经在跑，先 stop"}
        self.stop.clear()
        t = threading.Thread(target=self._play_loop, args=(float(duration), log, dry_run),
                             daemon=True)
        self.play_thread = t
        t.start()
        return {"playing": True, "duration": duration, "dry_run": dry_run,
                "log": log or ""}

    def _log(self, line: str, sink) -> None:
        self.last_log.append(line)
        if len(self.last_log) > 200:
            del self.last_log[:100]
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")

    def _play_loop(self, duration: float, log: str | None, dry_run: bool) -> None:
        sink = None
        if log:
            p = Path(log)
            if not p.is_absolute():
                p = ROOT / p
            p.parent.mkdir(parents=True, exist_ok=True)
            sink = p.open("a", encoding="utf-8", buffering=1)
        n = 0
        t0 = time.time()
        try:
            hwnd, _text, _cfg = self.hwnd()
            focus(hwnd)
            while not self.stop.is_set() and time.time() - t0 < duration:
                if emergency_stopped():
                    self._log(f"F12 急停，已跑 {n} 拍", sink)
                    break
                img, _info = grab(hwnd)
                fh, fw = img.shape[:2]
                hp = self_hp(img)
                bars = find_red_bars(img)
                bars.sort(key=lambda b: (b["cx"] - fw / 2) ** 2 + (b["cy"] - fh / 2) ** 2)
                hp_txt = "?" if hp["ratio"] is None else f"{hp['ratio']:.2f}"
                head = f"[{n:03d}] t={time.time() - t0:5.1f}s hp={hp_txt} 红条{len(bars)}"
                if hp["ratio"] is None:
                    self._log(head + " 读不到血量（已死/不在对局），本拍不动", sink)
                elif bars:
                    t = bars[0]
                    if not dry_run:
                        click_client(hwnd, t["cx"] / fw, (t["cy"] + 10) / fh, button="right",
                                     mode="raw")
                    dist = abs(t["cx"] - fw / 2)
                    face = ""
                    if dist < fw * 0.18 and hp["ratio"] >= LOW_HP and not dry_run:
                        press_key("w")
                        press_key("q")
                        face = " 贴脸放 wq"
                    self._log(head + f" 打 @({t['cx']:.0f},{t['cy']:.0f}) 距中 {dist:.0f}px{face}", sink)
                else:
                    if not dry_run:
                        click_client(hwnd, 0.62, 0.52, button="right")
                    self._log(head + " 没看到红条，往右推", sink)
                n += 1
                time.sleep(0.6)
        except Exception as e:  # 循环里出错要落日志，不能静默死掉
            self._log(f"对局循环异常退出：{type(e).__name__}: {e}", sink)
        finally:
            if sink:
                sink.close()
            self._log(f"对局循环结束，共 {n} 拍", None)


class Server:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, PORT))
        self.sock.listen(4)
        self.sock.settimeout(1.0)

    def serve_forever(self) -> None:
        print(f"bridge 在 {HOST}:{PORT} 上等命令（管理员={is_admin()}）", flush=True)
        while True:
            try:
                conn, _addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                f = conn.makefile("rwb")
                line = f.readline()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                    resp = self.dispatch(req)
                except Exception as e:
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                if resp.get("_quit"):
                    resp.pop("_quit")
                    f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                    f.flush()
                    return
                f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()

    def dispatch(self, req: dict) -> dict:
        cmd = (req.get("cmd") or "").lower()
        g = self.game
        if cmd == "ping":
            return {"ok": True, "admin": is_admin(), "port": PORT}
        if cmd == "quit":
            return {"ok": True, "_quit": True}
        if cmd == "state":
            return {"ok": True, **g.state()}
        if cmd == "shot" or cmd == "grab":
            return {"ok": True, **g.shot(req.get("path") or "tmp/frame.png")}
        if cmd == "click":
            return {"ok": True, **g.click(req.get("ui"), req.get("point"),
                                         (req.get("button") or "left").lower(),
                                         (req.get("mode") or "ui").lower(),
                                         float(req.get("hold") or 0.06))}
        if cmd == "clickr":  # 对局世界点击：走相对移动（raw input），大厅界面别用
            return {"ok": True, **g.click(None, req.get("point"),
                                         (req.get("button") or "right").lower(), "raw",
                                         float(req.get("hold") or 0.06))}
        if cmd == "rmove":
            move_rel(int(req.get("dx", 0)), int(req.get("dy", 0)))
            return {"ok": True, "dx": req.get("dx"), "dy": req.get("dy")}
        if cmd == "key":
            return {"ok": True, "vk": press_key(str(req.get("name"))), "name": req.get("name")}
        if cmd == "wait":
            time.sleep(float(req.get("seconds", 1)))
            return {"ok": True}
        if cmd == "play":
            return {"ok": True, **g.start_play(req.get("duration", 120),
                                              req.get("log"), dry_run=bool(req.get("dry_run")))}
        if cmd == "stop":
            g.stop.set()
            return {"ok": True, "stopped": True}
        return {"ok": False, "error": f"不认识的命令 {cmd!r}"}


def _start_elevated_hidden() -> int:
    """以管理员身份、隐藏控制台启动自己；UAC 只这一次。"""
    params = f'"{Path(__file__).resolve()}" --serve --hidden'
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params,
                                             str(ROOT), 0)
    if rc <= 32:
        print(f"提权失败（ShellExecute 返回 {rc}，1223=用户点了取消）")
        return 2
    print("已请求提权（隐藏窗口），UAC 弹窗里点「是」；之后不会再弹")
    return 0


def main(argv: list[str]) -> int:
    fix_console()
    enable_dpi_awareness()
    if "--serve" not in argv:
        print(__doc__)
        return 0
    if not is_admin():
        return _start_elevated_hidden()
    game = Game()
    srv = Server(game)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        game.stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
