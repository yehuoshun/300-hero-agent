"""回归集：拿固定语料给感知打分，改算法前后跑同一批图对比。

三项指标（见 skills/hero-screen-tuning）：
  1. 对局帧识别   —— 1280x800 的大厅图不该被当成对局
  2. 自身血量可读 —— 能读出的比例；读不出要能说清是「已死」还是「抠错」
  3. 单位检出数   —— 红条候选数（只做纵向对比，不设绝对值门槛）

用法：
    python tools/regress.py                     # 扫默认语料目录
    python tools/regress.py --dir <目录> --csv tmp/regress.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from perceive import self_hp  # noqa: E402
from play import find_red_bars  # noqa: E402

CORPUS = Path(r"E:\Game\JumpGame\300Hero\Screenshots")


def analyze(path: Path) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"file": path.name, "ok": False, "note": "读不到"}
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    battle = h == 1440 and w == 2560      # 对局＝全屏；大厅＝1280x800 窗口
    row = {"file": path.name, "ok": True, "size": f"{w}x{h}", "battle": battle,
           "hp": None, "note": "", "units": 0}
    if not battle:
        row["note"] = "非对局帧（大厅/菜单）"
        return row
    hp = self_hp(img)
    row["hp"] = hp["ratio"]
    row["note"] = hp["note"]
    row["units"] = len(find_red_bars(img))
    return row


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(CORPUS))
    ap.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    files = sorted(Path(a.dir).glob("*.bmp"))
    if not files:
        sys.exit(f"{a.dir} 里没有 .bmp")

    rows = [analyze(p) for p in files]
    battles = [r for r in rows if r.get("battle")]
    readable = [r for r in battles if r["hp"] is not None]
    dead = [r for r in battles if r["hp"] is None and "没有绿条" in r["note"]]
    suspicious = [r for r in battles if r["hp"] is None and r not in dead]
    units = [r["units"] for r in battles]

    print(f"语料 {len(rows)} 张：对局帧 {len(battles)}，非对局 {len(rows) - len(battles)}")
    print(f"自身血量：可读 {len(readable)}，已死/0 血 {len(dead)}，没读出来 {len(suspicious)}")
    if battles:
        print(f"红条候选数：min {min(units)} / median {sorted(units)[len(units)//2]} / max {max(units)}")
    for r in suspicious:
        print(f"  异常 {r['file']}: {r['note']}")
    if readable:
        vals = sorted(r["hp"] for r in readable)
        print(f"血量分布：{vals[0]:.2f} … {vals[-1]:.2f}")

    if a.csv:
        p = ROOT / a.csv
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["file", "size", "battle", "hp", "units", "note"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"明细：{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
