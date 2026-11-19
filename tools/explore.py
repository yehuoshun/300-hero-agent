"""探索客户端：把「一步一步试探游戏」写成脚本，一次跑完，省得人肉一条条敲 ctl。

为什么放在客户端而不是 bridge 里：bridge 一旦重启就要再弹一次 UAC（用户明确要求只弹一次），
所以「怎么探索」这类会反复改的逻辑都放这边，bridge 只当一只不会累的手。

用法（先把 bridge 起好，见 src/bridge.py 顶部注释）：

    python tools/explore.py frames --count 20 --interval 3 --prefix tmp/explore/obs
        每隔 N 秒抓一帧，存 <prefix>-000.png ...，用来观察阶段/界面自然变化

    python tools/explore.py walk --radius 0.18 --hold 1.4
        朝 8 个方向各点一次右键（走到位再抓图），存 tmp/explore/walk-<方向>-{before,after}.png

    python tools/explore.py keys --keys q,w,e,r,y,b,s,space --hold 1.0
        逐个按键，按键前后各抓一帧并算帧差，用来看这个键到底有什么效果

    python tools/explore.py click --at 0.5,0.5 --button right --shot tmp/explore/x.png
        点一下，顺便抓图

所有抓图都走 bridge（＝提权进程），脚本本身不碰游戏窗口。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 52341
ROOT = Path(__file__).resolve().parent.parent


def send(req: dict) -> dict:
    with socket.create_connection((HOST, PORT), timeout=30) as s:
        f = s.makefile("rwb")
        f.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        f.flush()
        line = f.readline()
    if not line:
        return {"ok": False, "error": "bridge 没回话"}
    return json.loads(line.decode("utf-8"))


def must(req: dict) -> dict:
    r = send(req)
    if not r.get("ok"):
        raise SystemExit(f"命令失败 {req.get('cmd')}: {r.get('error')}")
    return r


def shot(path: str) -> str:
    r = must({"cmd": "shot", "path": path})
    return r["path"]


def read_img(path: str | Path) -> np.ndarray:
    """cv2.imread 在中文路径下会返回 None，必须走 imdecode。"""
    buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到图 {path}")
    return img


def diff(a: str | Path, b: str | Path) -> dict:
    ia, ib = read_img(a), read_img(b)
    if ia.shape != ib.shape:
        return {"shape_diff": [list(ia.shape), list(ib.shape)]}
    d = cv2.absdiff(ia, ib).mean(axis=2)
    return {"mean": round(float(d.mean()), 2),
            "big_ratio": round(float((d > 30).mean()), 4)}


def scene(name: str, prefix: str = "tmp/explore") -> str:
    Path(ROOT / prefix).mkdir(parents=True, exist_ok=True)
    return f"{prefix}/{name}.png"


def cmd_frames(args) -> int:
    """定时抓帧；给了 --nav 就每次抓帧前先往那个方向点一下（边走边看）。"""
    nav = None
    if args.nav:
        nav = [float(v) for v in args.nav.split(",")]
    for i in range(args.count):
        if nav:
            must({"cmd": "click", "point": nav, "button": args.nav_button})
            time.sleep(0.5)
        p = shot(f"{args.prefix}-{i:03d}.png")
        st = must({"cmd": "state"})
        hp = st.get("hp")
        extra = f" 点了{nav}" if nav else ""
        print(f"[{i:03d}] {p} hp={hp} 红条={st.get('red_bars')}{extra}", flush=True)
        if i + 1 < args.count:
            time.sleep(args.interval)
    return 0


DIRS = {
    "up": (0.5, 0.34), "down": (0.5, 0.68),
    "left": (0.32, 0.51), "right": (0.68, 0.51),
    "up-left": (0.36, 0.38), "up-right": (0.64, 0.38),
    "down-left": (0.36, 0.64), "down-right": (0.64, 0.64),
}


def cmd_walk(args) -> int:
    """八个方向各走一次：走之前抓一帧、走之后抓一帧，作为相机/位移标定素材。"""
    cx, cy = 0.5, 0.51
    names = args.dirs.split(",") if args.dirs else list(DIRS)
    for name in names:
        if name not in DIRS:
            raise SystemExit(f"不认识的方向 {name!r}，可选：{','.join(DIRS)}")
        before = shot(scene(f"walk-{name}-before", args.prefix))
        tx, ty = DIRS[name]
        must({"cmd": "click", "point": [tx, ty], "button": "right"})
        time.sleep(args.hold)
        after = shot(scene(f"walk-{name}-after", args.prefix))
        d = diff(before, after)
        print(f"{name:10s} 点({tx:.2f},{ty:.2f}) 帧差={d}", flush=True)
        must({"cmd": "click", "point": [cx, cy], "button": "right"})
        time.sleep(0.8)
    return 0


def cmd_keys(args) -> int:
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    for k in keys:
        before = shot(scene(f"key-{k.replace(' ', '_')}-before", args.prefix))
        must({"cmd": "key", "name": k})
        time.sleep(args.hold)
        after = shot(scene(f"key-{k.replace(' ', '_')}-after", args.prefix))
        d = diff(before, after)
        print(f"key {k:6s} 帧差={d}", flush=True)
    return 0


def cmd_click(args) -> int:
    x, y = (float(v) for v in args.at.split(","))
    r = must({"cmd": "click", "point": [x, y], "button": args.button})
    print(f"clicked {r['rel']} -> screen {r['screen']}", flush=True)
    if args.shot:
        print(shot(args.shot), flush=True)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)

    p = sub.add_parser("frames")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--prefix", default="tmp/explore/obs")
    p.add_argument("--nav", default="", help="每拍前先点这里，例如 0.68,0.50")
    p.add_argument("--nav-button", default="right")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("walk")
    p.add_argument("--hold", type=float, default=1.4)
    p.add_argument("--dirs", default="")
    p.add_argument("--radius", type=float, default=0.18)   # 目前坐标写死在 DIRS 里
    p.add_argument("--prefix", default="tmp/explore")
    p.set_defaults(func=cmd_walk)

    p = sub.add_parser("keys")
    p.add_argument("--keys", default="q,w,e,r,y,b,s,space")
    p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--prefix", default="tmp/explore")
    p.set_defaults(func=cmd_keys)

    p = sub.add_parser("click")
    p.add_argument("--at", default="0.5,0.5")
    p.add_argument("--button", default="left")
    p.add_argument("--shot", default="")
    p.set_defaults(func=cmd_click)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
