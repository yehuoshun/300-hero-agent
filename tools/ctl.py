"""给 bridge 发命令的客户端（非提权侧用）。

例：
    python tools/ctl.py ping
    python tools/ctl.py state
    python tools/ctl.py shot tmp/a.png
    python tools/ctl.py click ui hall_attack
    python tools/ctl.py click at 0.5 0.52 right
    python tools/ctl.py key w
    python tools/ctl.py play 90 tmp/play.log
    python tools/ctl.py stop
    python tools/ctl.py quit

回包一律是 JSON 一行，原样打印；ok=false 时退出码 1，方便脚本判。
"""
from __future__ import annotations

import json
import socket
import sys

HOST, PORT = "127.0.0.1", 52341


def build(argv: list[str]) -> dict:
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0].lower(), argv[1:]
    if cmd in ("ping", "state", "stop", "quit"):
        return {"cmd": cmd}
    if cmd == "shot" or cmd == "grab":
        return {"cmd": "shot", "path": rest[0] if rest else "tmp/frame.png"}
    if cmd == "click":
        # click ui <键>            / click at <x> <y> [left|right]
        if rest and rest[0] == "ui":
            if len(rest) < 2:
                raise SystemExit("click ui 后面要跟 config.ui 里的键名")
            return {"cmd": "click", "ui": rest[1]}
        if rest and rest[0] == "at":
            if len(rest) < 3:
                raise SystemExit("click at 后面要跟 x y [left|right]（相对比例 0~1）")
            button = rest[3] if len(rest) > 3 else "left"
            return {"cmd": "click", "point": [float(rest[1]), float(rest[2])],
                    "button": button}
        raise SystemExit("click 只支持：click ui <键名> / click at <x> <y> [left|right]")
    if cmd == "clickr":
        # clickr <x> <y> [left|right]：对局内世界点击，走相对移动（raw input）
        if len(rest) < 2:
            raise SystemExit("clickr 后面要跟 x y [left|right]（相对比例 0~1）")
        button = rest[2] if len(rest) > 2 else "right"
        return {"cmd": "clickr", "point": [float(rest[0]), float(rest[1])], "button": button}
    if cmd == "rmove":
        if len(rest) < 2:
            raise SystemExit("rmove 后面要跟 dx dy（像素）")
        return {"cmd": "rmove", "dx": int(rest[0]), "dy": int(rest[1])}
    if cmd == "key":
        if not rest:
            raise SystemExit("key 后面要跟键名（a/s/q/w/e/r/b/y/esc/enter…）")
        return {"cmd": "key", "name": rest[0]}
    if cmd == "wait":
        return {"cmd": "wait", "seconds": float(rest[0]) if rest else 1.0}
    if cmd == "play":
        duration = float(rest[0]) if rest else 120.0
        log = rest[1] if len(rest) > 1 else "tmp/play.log"
        return {"cmd": "play", "duration": duration, "log": log}
    raise SystemExit(f"不认识的命令 {cmd!r}，见文件头用法")


def main(argv: list[str]) -> int:
    req = build(argv)
    try:
        with socket.create_connection((HOST, PORT), timeout=60) as s:
            s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except ConnectionRefusedError:
        print(json.dumps({"ok": False, "error": "bridge 没在跑：先 python src/bridge.py --serve --hidden"},
                         ensure_ascii=False))
        return 2
    text = buf.decode("utf-8", "replace").strip()
    print(text)
    try:
        return 0 if json.loads(text).get("ok") else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
