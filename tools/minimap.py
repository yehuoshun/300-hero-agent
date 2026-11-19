"""小地图（右上角）识别：矩形定位、自家标记、图标分类、小地图像素→归一化世界坐标。

纯离线工具：只吃已有 PNG，不抓屏、不发输入、不联网、不 import mss/bridge/play/capture。

所有常量都是 2026-11-19 在 2560x1440 真实对局帧上逐像素量出来的（探测脚本 tmp/mm-*.py），
不是猜的；每处判据旁边写清量法与已知误检源。

用法：
    python tools/minimap.py --selftest tmp/m-a.png tmp/k3-move.png
    python tools/minimap.py --overlay tmp/m-a.png tmp/mm-overlay-a.png

坐标约定：find_self_marker / detect_icons 返回的是**小地图内局部像素**（左上角 = 矩形左上角）；
world_from_minimap 也吃局部像素，输出 0~1 归一化坐标。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 实测常量

# 小地图矩形相对整帧的比例（左/上），以及矩形尺寸比例。实测像素 (2303,1180)-(2557,1432)。
RECT_RATIO = (0.89961, 0.81944)                 # 左上角
SIZE_RATIO = (0.09922, 0.17500)                 # 宽 254 / 高 252（2560x1440 下）

# 地图内容区（去掉顶边框亮线与底部队友头像行），相对矩形自身的局部比例。
# 实测：顶亮线 y=1180；头像行从 y≈1405 起抢占地图像素（p0 在 x2550..2556 整列都是头像）。
# ⚠ 这里 0.861（局部 y≈217）不是「地图内容的真实下界」，只是「干净检出区」：
#   把下界放到 0.985 / 1.0 实测 ally/enemy 数量**完全不变**，只多出底部头像行与贴底残片（others +1~4）；
#   贴下边框被截断的塔（局部 (67,245)/(118,245) 9x7、p 系列的蓝塔 (217,236) 8x9）
#   因形状只剩一半、且位置随对局固定，未纳入塔统计 —— 这是已知漏检（见 tmp/mm-final.py 输出）。
CONTENT_LOCAL_RATIO = (0.004, 0.012, 0.996, 0.861)

# 右上角白色亮线 UI 面板（实测 (2507,1185)-(2556,1220)），里面有固定装饰（亮绿小环 + 蓝圆图标），
# 不排除会把 (2539,1198) 当英雄标记/塔。相对矩形自身局部坐标。
HUD_PANEL_LOCAL = (204, 5, 253, 40)
# 跨 19 帧位置完全不变的 HUD 装饰（绿环形状），必须从图标统计里排除：
#   (199.5,134.5) —— p 系列每帧都在的假绿环；(235.9,18.5) 落在上面的面板里，已被面板矩形排除。
FIXED_DECO_LOCAL = ((199.5, 134.5),)

# 世界尺度标定：**未确定**。没有 mm_px/world_px 这个常量，不要换算真实距离。证据见 CALIBRATION_NOTE。
CALIBRATION_NOTE = (
    "**世界尺度未确定 —— mm_px/world_px 的 x、y 两个分量都给不出数，代码里不提供这个常量。**\n"
    "【否证 1：任务书给的 m-a→m-b (dx=+270, dy=-135) 不成立】\n"
    "  全帧 SIFT+RANSAC（estimateAffinePartial2D / RANSAC 3px）得 dx=-0.03 dy=-0.08 scale=1.0001；"
    "全帧 phaseCorrelate 得 (+0.02,-0.03) resp 0.383；小地图区 absdiff mean 只有 0.494。"
    "→ 这两帧之间相机没动，差异是单位动画/特效（全帧 absdiff mean 16.51 集中在单位身上）。\n"
    "【否证 2：11 组相邻帧里**没有任何一组**相机平移过】\n"
    "  全帧 SIFT+RANSAC(3px)：m-a→m-b (-0.03,-0.08) / m-c→m-d / m-e→m-f / p0→p1 / p1→p2 / "
    "k1-y→k2-space / k2-space→k3-move / k3-move→k4-later / k0→k1-y **全部 |d|<=0.1 px、scale 1.0000**"
    "（k0→k1-y：2043/2911 inliers，dx=-0.05 dy=+0.06 scale=1.0000）。\n"
    "  全帧 phaseCorrelate：9 组 |d|<=0.1；剩下两组是低响应伪峰（m-b→m-c dx=-260.5 但 resp=0.0143；"
    "k0→k1-y dy=+10.5 但 resp=0.196）。\n"
    "  3 个 160x160 模板 TM_CCOEFF_NORMED ±420px 搜索、中心块 ±380px 暴力最小 MAD：最优解都是 (0,0)，"
    "或增益只有 4/40 的噪声级（m-a→m-b (+232,0) MAD 36.29 vs (0,0) 40.32）。\n"
    "【作废：之前记过的 k0→k1-y (dx=+10.02, dy=+14.25)】**\n"
    "  那是 inliers 只有 238/421(57%) 的拟合；用同一对帧做完整 RANSAC 复核（2043/2911 inliers）得 (-0.05,+0.06)。"
    "所以那组数以及由它推出的 MM_PX_PER_WORLD_PX=(0.220,0.119) **全部作废**。\n"
    "【更根本的原因：小地图不是相机窗口，是整张世界地图的俯瞰图】\n"
    "  铁证：9 座己方塔里有 5 座坐标在 14/14 个已渲染帧中**逐帧完全相同**"
    "（(144,15)(183,89)(153,111)(251,124)(251,195)），敌方塔同理；地形/水域大块也位置不变。"
    "整张图固定 → **相机怎么平移都不会让小地图内容平移**，所以「相机位移 ÷ 小地图位移」这个模型本身不成立，"
    "不是素材不够。\n"
    "【两个观测互相矛盾，说明至少一条通道不是忠实的位移读数】\n"
    "  k0→k1-y 自家白框中心 (202,186)→(199,184) 即 Δmm=(-3,-2)，k2-space→k3-move Δmm=(-5,-3)，"
    "p0→p1 Δmm=(-10,-6)，p1→p2 Δmm=(-13,+3)；可同一时刻背景是像素级的 0 位移。"
    "若相机死锁跟人，Δmm 非零就该伴随背景同量平移；若相机有死区（只在英雄出死区才回中），"
    "则背景不动而 Δmm 动 —— 两种解释都能自洽，但都无法给出比例。\n"
    "【唯一能确定的】\n"
    "  小地图坐标是**世界坐标的线性映射**（同一局内塔坐标逐帧不动，说明没有缩放/旋转/平移漂移），"
    "所以 world_from_minimap 给出的是「在图上的归一化位置」，可作为稳定世界坐标用；"
    "但它离真实世界单位（米/屏幕像素）的换算比例**未确定**。要标定需要："
    "①一对背景确实平移过（>=3px）且自家标记两帧都命中的帧；或②同时测出「英雄在屏幕上的位移」"
    "（英雄身体/名牌）与「小地图标记位移」，用 Δmm/Δscreen —— 现有 19 帧都没做这件事。"
)

# 两个跨 19 帧位置完全不变的暗绿环状标记（疑野怪营地），相对内容框的比例。
# 色值与地形重叠（H≈30 S≈90 V≈75，与周边草地同色），HSV 分不开，所以用「位置锁定 + 环状」双判据。
CAMP_HINT_RATIOS = ((0.692, 0.141), (0.814, 0.375))
CAMP_HINT_TOL = 6.0                             # 小地图像素

# ---------------------------------------------------------------- 地形色域（实测）
# 对内容框做 k-means(k=7) + 定点采样（tmp/mm-terrain.py、tmp/mm-water.py、tmp/mm-region.py）：
#   河道/水体  H 85~120 S 40~110 V 50~130，中位 HSV(95,60,87)，占 5~7%；左上有一块固定大湖
#              (局部 54,55,69x68, a=1309)，跨帧位置完全一致。
#              注意：水体是**低饱和**青灰（S≈60），而蓝塔是高饱和蓝（S≈190 V≈190）——靠 S 就能分开。
#   草地/地表  H 16~40 S 40~160 V 70~133，占约 30%（橄榄绿，与野怪营地同色）。
#   深色墙体/描边/阴影  V<55（H 0~60 S 0~255），占 32~37% —— 地图轮廓、建筑描边都在这档。
#              ⚠ 这一档**不能**用来判定「地图是否渲染」：黑屏帧整块都落进来（实测占比 0.88 > 正常帧 0.76）。
#   亮灰/米黄道路  H 13~26 S 36~120 V 134~224，占 6~15%。
#   图标/高光  V>200 且 S<100 或 S>170，占 2~4%。
# 「地形 vs 图标」的可分性：塔/头像/营地全部是 H 落在窄带且 S>=140 的高饱和块，
# 地形最高饱和出现在深色墙体（V<55），所以 **S>=140 & V>=90** 已经能把图标和地形分开；
# 唯一与地形同色的是野怪营地（H≈30 S≈90 V≈75），只能靠位置锁定。
TERRAIN_HSV = {
    "water": ((85, 120), (40, 110), (50, 130)),
    "grass": ((16, 40), (40, 160), (70, 133)),
    "dark": ((0, 60), (0, 255), (0, 55)),
    "road": ((13, 26), (36, 120), (134, 224)),
}
# 地图内容是否真的渲染出来了：按 grass|water|road 统计（不含 dark 带，dark 带会把黑屏算成 0.88 反过来）。
# 实测：正常帧 0.31~0.42，黑屏/另一套界面帧 0.002~0.003（m-g / t0 / t2-walk / t3-walk / n1）。
# 低于这个阈值时 detect_icons 返回空是正常的，不是判据失效。
CONTENT_TERRAIN_MIN = 0.20


# ---------------------------------------------------------------- 读图

def read_frame(path: str | Path) -> np.ndarray:
    buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不出图：{path}")
    return img


def write_frame(path: str | Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"编码失败：{path}")
    Path(path).write_bytes(buf.tobytes())


def _hsv(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 0].astype(np.int16),
            hsv[:, :, 1].astype(np.int16),
            hsv[:, :, 2].astype(np.int16))


# ---------------------------------------------------------------- 1. 矩形

def _frameline_mask(frame: np.ndarray) -> np.ndarray:
    """小地图边框的「亮细线」：实测 H 不定、S<=140、V>=150（m-a 上 y=1180 整行 244 px）。"""
    _, s, v = _hsv(frame)
    return ((v >= 150) & (s <= 140)).astype(np.uint8)


def minimap_rect(frame: np.ndarray) -> tuple[int, int, int, int]:
    """返回小地图矩形 (x0, y0, x1, y1)，x1/y1 不含。

    判据（可复现）：边框是两条**亮细线**（V>=150 且 S<=140）。
      ① 上边 = 在 y∈[比例±2%] 内「亮线像素最多」的那一行（m-a/k0/p0 实测 244~245 px，稳）；
      ② 左边 = 在 x∈[比例±1.5%] 内「亮线像素最多」的那一列（实测 x=2303，y 1183..1396 共 211~219 px）；
      ③ 右/下边不靠亮线（实测右边缘是暗线、下边缘被队友头像行占），改用固定尺寸
         (254,252)（= 0.09922W × 0.175H）从左上角推出。
    吸附失败（亮线点数不够）就退回纯比例值。
    """
    H, W = frame.shape[:2]
    r = RECT_RATIO
    x0 = int(round(r[0] * W))
    y0 = int(round(r[1] * H))
    mw = int(round(SIZE_RATIO[0] * W))
    mh = int(round(SIZE_RATIO[1] * H))

    line = _frameline_mask(frame)

    # ① 上边
    lo, hi = max(0, y0 - int(0.02 * H)), min(H - 1, y0 + int(0.02 * H))
    if hi > lo:
        span = line[lo:hi + 1, x0:x0 + mw]
        counts = span.sum(axis=1)
        if counts.max() >= 80:
            y0 = lo + int(counts.argmax())

    # ② 左边
    lo, hi = max(0, x0 - int(0.015 * W)), min(W - 1, x0 + int(0.015 * W))
    if hi > lo:
        span = line[y0:y0 + mh, lo:hi + 1]
        counts = span.sum(axis=0)
        if counts.max() >= 60:
            x0 = lo + int(counts.argmax())

    x1 = min(W, x0 + mw)
    y1 = min(H, y0 + mh)
    return x0, y0, x1, y1


def _content_local(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    c = CONTENT_LOCAL_RATIO
    return (int(round(c[0] * w)), int(round(c[1] * h)),
            int(round(c[2] * w)), int(round(c[3] * h)))


def _content_box(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """『地图内容』全局框（去掉顶边框和底部头像行）。"""
    x0, y0, _, _ = rect
    cx0, cy0, cx1, cy1 = _content_local(rect)
    return x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1


# ---------------------------------------------------------------- 2. 自家标记

# 白框实测（raw 掩码 S<60 & V>190，不做形态学，否则 fill 会被拉到 0.71）：
#   k0 376px 25x26 fill .578 perim .396 / k1-y 376 / k2-space 376 / k3-move 376 /
#   k4-later 366 25x26 .563 .359 / p0 376 / p1 344 25x26 .529 .341 / p2 374 25x25 .598 .455。
#   另有 24x25 area=305 fill .51 perim .34 的**更小一档**：tmp/raw/y0.png (198,84)、x9.png (163,92)，
#   只比 320 的下界小 15 px —— 早先 SELF_AREA 下界 320 会把这两帧判成 None（实测已借放宽版验明）。
SELF_AREA = (290, 400)
SELF_WH = (22, 29)
SELF_FILL = (0.45, 0.66)
SELF_PERIM = 0.30


def _perimeter_coverage(comp: np.ndarray, thickness: int = 3) -> float:
    """连通域在自身 bbox 外圈 3 px 带内的覆盖率 —— 方框高、圆斑低。"""
    h, w = comp.shape
    band = np.zeros_like(comp)
    t = min(thickness, max(1, min(h, w) // 4))
    band[:t, :] = True
    band[-t:, :] = True
    band[:, :t] = True
    band[:, -t:] = True
    area = int(band.sum())
    return float(comp[band].sum()) / area if area else 0.0


def find_self_marker(frame: np.ndarray) -> dict | None:
    """找自家英雄标记（白色空心方框包住英雄头像+绿环），返回小地图内局部像素中心。

    判据：S<60 且 V>190 的连通域，bbox 23~28 px 见方、面积 320~400、填充比 0.48~0.64、
    外圈 3 px 覆盖率>=0.30；并排除右上 HUD 面板与底部头像行。
    跨 k0/k1-y/k2-space/k3-move/k4-later/p0/p1/p2 共 8 帧命中；
    m-a..m-g / t0 / t2-walk / t3-walk / n1 这 11 帧里**没有白框**（另一套界面状态），返回 None 是正确行为。
    """
    rect = minimap_rect(frame)
    x0, y0, x1, y1 = rect
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    _, s, v = _hsv(sub)
    white = ((s < 60) & (v > 190)).astype(np.uint8)

    cx0, cy0, cx1, cy1 = _content_local(rect)
    keep = np.zeros_like(white)
    keep[cy0:cy1, cx0:cx1] = 1
    white &= keep
    px0, py0, px1, py1 = HUD_PANEL_LOCAL
    white[py0:py1, px0:px1] = 0

    n, lab, st, cen = cv2.connectedComponentsWithStats(white, 8)
    best = None
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if not (SELF_AREA[0] <= a <= SELF_AREA[1]):
            continue
        w = int(st[i, cv2.CC_STAT_WIDTH])
        h = int(st[i, cv2.CC_STAT_HEIGHT])
        if not (SELF_WH[0] <= w <= SELF_WH[1] and SELF_WH[0] <= h <= SELF_WH[1]):
            continue
        fill = a / float(w * h)
        if not (SELF_FILL[0] <= fill <= SELF_FILL[1]):
            continue
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        perim = _perimeter_coverage(lab[by:by + h, bx:bx + w] == i)
        if perim < SELF_PERIM:
            continue
        conf = float(min(1.0, 0.55 * perim + 0.45 * max(0.0, 1.0 - abs(fill - 0.57) / 0.14)))
        if best is None or conf > best["conf"]:
            best = {"x": int(round(cen[i][0])), "y": int(round(cen[i][1])), "conf": round(conf, 3)}
    return best


def content_terrain_fraction(frame: np.ndarray) -> float:
    """内容框里「像地图地形」的像素占比。用来区分「检测失败」和「地图根本没渲染」。

    判据 = 草地带 ∪ 水体带 ∪ 道路带（**不含 dark 带**，见 TERRAIN_HSV 的 docstring）。
    ⚠ 实测教训：如果把 dark 带（V<55）也算进来，黑屏帧反而 0.80~0.88 > 正常帧 0.72~0.76，
      判据完全反转 —— 因为黑屏就是「全暗」。所以这里只统计**有彩色的地表**。
    实测：正常对局帧 0.31~0.42（m 系列 0.42、k 系列 0.32~0.34、p 系列 0.31）；
    黑屏/另一套界面帧 0.002~0.003（m-g / t0 / t2-walk / t3-walk / n1）。
    """
    rect = minimap_rect(frame)
    cx0, cy0, cx1, cy1 = _content_box(rect)
    sub = frame[cy0:cy1, cx0:cx1]
    if sub.size == 0:
        return 0.0
    h, s, v = _hsv(sub)
    m = np.zeros(h.shape, bool)
    for key in ("grass", "water", "road"):
        (hlo, hhi), (slo, shi), (vlo, vhi) = TERRAIN_HSV[key]
        m |= (h >= hlo) & (h <= hhi) & (s >= slo) & (s <= shi) & (v >= vlo) & (v <= vhi)
    return float(m.mean())


# ---------------------------------------------------------------- 3. 图标

def _blobs(mask: np.ndarray, min_area: int = 8, close: int = 0) -> list[dict]:
    if close > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               np.ones((close, close), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        w = int(st[i, cv2.CC_STAT_WIDTH])
        h = int(st[i, cv2.CC_STAT_HEIGHT])
        out.append({"cx": float(cen[i][0]), "cy": float(cen[i][1]), "w": w, "h": h,
                    "a": a, "fill": a / float(w * h), "x": int(st[i, cv2.CC_STAT_LEFT]),
                    "y": int(st[i, cv2.CC_STAT_TOP])})
    return out


def detect_icons(frame: np.ndarray) -> dict:
    """小地图内图标分类。坐标都是小地图内局部像素。

    判据（全部实测，误检源见括号）：
      己方塔   H 95~115 S>=140 V>=140，9x9 实心菱形（a 35~100）；贴右边缘被截断的按 4~8 宽放行。
               （误检：右上 HUD 面板里同款蓝图标 → 用面板矩形状排除）
      敌方塔   H<=10 或 >=170 S>=140 V>=90，5~12 宽 7~16 高、面积>=22、fill>=0.40。
               （菱形底座 9x4 fill .72 被高>=7 挡掉；红色圆圈 10x10 fill .25 被 fill 挡掉）
      野怪营地 暗橄榄环 H 25~45 S 70~155 V 55~110 + 位置锁定（跨 19 帧不变的 2 个位置）。
      其他     yellow_icon(皇冠/火焰) / hero_ring(绿环=英雄标记) / red_dot / red_ring / blue_round / green_dot
    """
    rect = minimap_rect(frame)
    x0, y0, x1, y1 = rect
    sub = frame[y0:y1, x0:x1]
    out = {"ally_towers": [], "enemy_towers": [], "camps": [], "others": []}
    if sub.size == 0:
        return out

    h, s, v = _hsv(sub)
    hh, ww = h.shape
    cx0, cy0, cx1, cy1 = _content_local(rect)
    inside = np.zeros((hh, ww), bool)
    inside[cy0:cy1, cx0:cx1] = True
    px0, py0, px1, py1 = HUD_PANEL_LOCAL
    panel = np.zeros((hh, ww), bool)
    panel[py0:py1, px0:px1] = True
    usable = inside & ~panel

    def ok(cx: float, cy: float) -> bool:
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= iy < hh and 0 <= ix < ww):
            return False
        return bool(usable[iy, ix])

    def add(store: str, kind: str, b: dict, conf: float) -> None:
        if not ok(b["cx"], b["cy"]):
            return
        out[store].append({"x": int(round(b["cx"])), "y": int(round(b["cy"])),
                           "kind": kind, "conf": round(float(conf), 2)})

    # --- 己方塔（蓝菱形 + 蓝梯形石座）：实测稳定形态 9x15 a=69 fill .51（5 帧逐帧同坐标）
    #     误检源：右上 HUD 面板里两个 10x10 fill .76 的蓝双环 → 用 fill<=0.70 挡掉；
    #     大块 18x24 fill .32 / 21x9 / 13x14 属光效 → 用 w/h 上界挡掉。
    blue = ((h >= 95) & (h <= 115) & (s >= 140) & (v >= 140)).astype(np.uint8)
    for b in _blobs(blue, 16, close=3):
        if (54 <= b["a"] <= 80 and 6 <= b["w"] <= 11 and 13 <= b["h"] <= 17
                and b["fill"] <= 0.70):
            add("ally_towers", "ally_tower", b, min(1.0, b["a"] / 69.0))
        elif (b["x"] + b["w"] >= ww - 4 and 5 <= b["w"] <= 8 and 9 <= b["h"] <= 16
              and 50 <= b["a"] <= 72 and b["fill"] <= 0.72):   # 贴右边框被截断的塔（实测 4 座）
            add("ally_towers", "ally_tower_edge", b, 0.60)
        elif 20 <= b["a"] <= 110 and 8 <= b["w"] <= 13 and 8 <= b["h"] <= 14:
            add("others", "blue_round", b, 0.45)

    # --- 敌方塔（红菱形 + 红梯形底座）：实测 9x15 a=69 fill .51，贴左边框的并成 7x17/9x18 fill .44
    #     误检源：红色圆圈图标 10x10 fill .76（h=10 被 h>=14 挡掉）；24x24/25x24/44x40 光效（w 上界挡掉）
    red = (((h <= 10) | (h >= 170)) & (s >= 140) & (v >= 90)).astype(np.uint8)
    for b in _blobs(red, 9, close=3):
        if (50 <= b["a"] <= 90 and 6 <= b["w"] <= 13 and 14 <= b["h"] <= 21
                and 0.40 <= b["fill"] <= 0.62):
            add("enemy_towers", "enemy_tower", b, min(1.0, b["a"] / 69.0))
        elif b["a"] <= 20 and b["w"] <= 6 and b["h"] <= 6:
            add("others", "red_dot", b, 0.35)
        elif 60 <= b["a"] <= 90 and 8 <= b["w"] <= 13 and 8 <= b["h"] <= 11:
            add("others", "red_round", b, 0.50)      # 红圆圈图标（实测 10x10 fill .76~.77）

    # --- 野怪营地：暗橄榄纹理簇 + 位置锁定（2 个跨 19 帧坐标完全不变的位置）
    #     实测：营地与草地同色（H≈31 S≈95 V≈75），HSV 分不开，也不能做连通域形状判据
    #     （整片草地是一个大连通域）；可靠的是「位置恒定」+ 该处确有一团 26x27~29x29 橄榄纹理。
    #     → 只在锁定位置开 29x29 窗，验证窗口内橄榄像素占比与最大连通块尺寸。
    olive = ((h >= 25) & (h <= 45) & (s >= 88) & (s <= 175)
             & (v >= 50) & (v <= 115)).astype(np.uint8)
    for rx, ry in CAMP_HINT_RATIOS:
        kx = int(round(cx0 + rx * (cx1 - cx0)))
        ky = int(round(cy0 + ry * (cy1 - cy0)))
        if not (14 <= kx < ww - 14 and 14 <= ky < hh - 14):
            continue
        win = olive[ky - 14:ky + 15, kx - 14:kx + 15]
        frac = float(win.mean())
        n, _, st, _ = cv2.connectedComponentsWithStats(win, 8)
        big = [(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA])
               for i in range(1, n)]
        big = max(big, default=(0, 0, 0), key=lambda t: t[2])
        if 0.15 <= frac <= 0.70 and 16 <= big[0] <= 32 and 16 <= big[1] <= 32 and big[2] >= 80:
            out["camps"].append({"x": kx, "y": ky, "kind": "camp_guess", "conf": 0.45})

    # --- 黄色火焰/皇冠：实测只有一块是稳定的 11x13 a=93 fill .65（局部 (188,190)，全球 (2491,1369)）
    #     误检源：p 系列恒定的 8x12 fill .48 装饰（局部 (175,184)/(186,184)）→ 用 a>=60 挡掉；
    #     18x16 fill .37 / 16x24 / 25x23 光效 → w/h 上界挡掉。语义（营地？灵魂？）未确定。
    yellow = ((h >= 18) & (h <= 35) & (s >= 150) & (v >= 140)).astype(np.uint8)
    for b in _blobs(yellow, 40, close=5):
        if (60 <= b["a"] <= 130 and 10 <= b["w"] <= 14 and 10 <= b["h"] <= 16
                and b["fill"] >= 0.50):
            add("others", "yellow_icon", b, min(1.0, b["a"] / 93.0))

    # --- 英雄头像（绿环圆盘）= 小地图上的英雄标记，自家那颗被白框圈住。见 find_self_marker。
    #     判据：绿环 H45~90 S>=140 V>=100，close5；完整环 24x24~40x40（直径 30~34），fill<=0.45。
    #     19 帧实测（内容框内）：m-a 1 / m-b 1 / m-c 1 / m-d 1 / m-e 2 / m-f 1 / k0 3 / k1-y 3 /
    #     k2-space 3 / k3-move 3 / k4-later 1 / p0 4 / p1 1 / p2 2；空渲染帧（m-g/t0/t2-walk/t3-walk/n1）0。
    #     环色**与敌我无关**：14 帧内所有环像素中位 HSV 都是 H57~59 S202~226 V183~195，
    #     环外 annulus 的红色像素占比全为 0.00 → 「红环=敌方英雄」没有证据，不实现。
    #     误检源：右上 HUD 面板里的固定绿环（已在 panel 内）；p 系列恒定 (199.5,134.5) 的假环（FIXED_DECO_LOCAL）。
    ring = ((h >= 45) & (h <= 90) & (s >= 140) & (v >= 100)).astype(np.uint8)
    for b in _blobs(ring, 45, close=5):
        if 24 <= b["w"] <= 40 and 24 <= b["h"] <= 40 and b["fill"] <= 0.45:
            if any(abs(b["cx"] - fx) <= 4 and abs(b["cy"] - fy) <= 4
                   for fx, fy in FIXED_DECO_LOCAL):     # 跨帧恒定的 HUD 装饰，不是世界图标
                continue
            add("others", "hero", b, 0.5)

    # --- 贴下边框被截断的塔（内容框下界以外，所以绕过 ok() 单独判定）
    #     实测条件：y+h >= hh-4、6<=w<=14、4<=h<=11、a>=28、0.42<=fill<=0.72。
    #     m-a..m-g / t0 / t2-walk / t3-walk / n1 = 0 个（m 系列底部被队友头像行盖住）；
    #     k0/k1-y/k2-space/k3-move/k4-later/p0/p1/p2 **每帧 2~3 个**，坐标恒定：
    #     (71,248) 与 (122,248) 8 帧全有；(193,246) 除 p0 外都有（p0 该处被挡住）。
    #     因为只剩半个菱形、且位置跨帧完全固定，**不并入 enemy_towers**，只报 others 的 edge 类。
    for b in _blobs(red, 8, close=3):
        if (b["y"] + b["h"] >= hh - 4 and 6 <= b["w"] <= 14 and 4 <= b["h"] <= 11
                and b["a"] >= 28 and 0.42 <= b["fill"] <= 0.72):
            out["others"].append({"x": int(round(b["cx"])), "y": int(round(b["cy"])),
                                  "kind": "enemy_tower_edge_bottom", "conf": 0.40})

    # --- 亮绿小点（地形上的小标记，可靠度低）
    dot = ((h >= 40) & (h <= 80) & (s >= 170) & (v >= 130)).astype(np.uint8)
    for b in _blobs(dot, 14):
        if b["a"] <= 40 and b["w"] <= 9 and b["h"] <= 9:
            add("others", "green_dot", b, 0.25)

    return out


# ---------------------------------------------------------------- 4. 坐标换算

def world_from_minimap(mx: float, my: float) -> tuple[float, float]:
    """小地图内局部像素 → 归一化世界坐标 (u, v)，0~1。

    映射 = 「把整张小地图矩形线性拉伸到 0..1」：u/v 原点 = 矩形左上角（地图边框），
    1.0 = 矩形右下角。**不是**按「干净检出区」算 —— 内容框的 y1=0.861 只是为了躲底部队友头像行，
    拿它当 1.0 会把 v 整体放大 1/0.861，是错的。

    为什么可以把它当世界坐标：塔/水域/地形在小地图上逐帧不动（14/14 帧同坐标），
    说明小地图是「整张世界地图的固定俯瞰图」，与世界坐标是同一套线性坐标，跨帧不漂。
    ⚠ 未确定的两点（见 CALIBRATION_NOTE）：
      ① 矩形边框是否严格等于世界可玩边界（可能带 1~4px 留白/描边）；
      ② 与真实单位（米 / 屏幕像素）的换算比例，没有任何可靠数据。
    """
    w, h = minimap_rect_size()
    u = float(mx) / max(1.0, float(w))
    v = float(my) / max(1.0, float(h))
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def minimap_rect_size(frame_shape: tuple[int, int] = (1440, 2560)) -> tuple[int, int]:
    """矩形宽高（像素），按 SIZE_RATIO 从帧尺寸推。给 world_from_minimap 用。"""
    H, W = frame_shape[:2]
    return (int(round(SIZE_RATIO[0] * W)), int(round(SIZE_RATIO[1] * H)))


# ---------------------------------------------------------------- 5. CLI

_COLORS = {
    "ally_tower": (255, 200, 0), "ally_tower_edge": (255, 160, 0),
    "enemy_tower": (0, 0, 255), "camp_guess": (0, 255, 0),
    "hero": (0, 255, 128), "yellow_icon": (0, 200, 255),
    "red_dot": (60, 60, 255), "red_round": (120, 0, 255),
    "blue_round": (255, 120, 0), "green_dot": (120, 255, 120),
    "enemy_tower_edge_bottom": (0, 80, 255),
}
_TAGS = {
    "ally_tower": "A", "ally_tower_edge": "a", "enemy_tower": "E",
    "enemy_tower_edge_bottom": "e", "camp_guess": "C", "hero": "H",
    "yellow_icon": "Y", "red_dot": "r", "red_round": "R",
    "blue_round": "B", "green_dot": "g",
}


def _draw(frame: np.ndarray, rect, self_m, icons) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = rect
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
    cx0, cy0, cx1, cy1 = _content_box(rect)
    cv2.rectangle(out, (cx0, cy0), (cx1 - 1, cy1 - 1), (255, 128, 0), 1)

    def mark(gx, gy, color, rad, tag):
        cv2.circle(out, (gx, gy), rad, color, 1)
        cv2.line(out, (gx - rad - 4, gy), (gx + rad + 4, gy), color, 1)
        cv2.line(out, (gx, gy - rad - 4), (gx, gy + rad + 4), color, 1)
        cv2.putText(out, tag, (gx + rad + 2, gy - rad - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 0, 0), 3)
        cv2.putText(out, tag, (gx + rad + 2, gy - rad - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1)

    if self_m:
        gx, gy = x0 + self_m["x"], y0 + self_m["y"]
        cv2.rectangle(out, (gx - 16, gy - 16), (gx + 16, gy + 16), (0, 255, 255), 1)
        mark(gx, gy, (0, 255, 255), 12, "SELF")

    for name in ("ally_towers", "enemy_towers", "camps", "others"):
        for it in icons.get(name, []):
            col = _COLORS.get(it["kind"], (255, 0, 255))
            tag = _TAGS.get(it["kind"], "?")
            mark(x0 + it["x"], y0 + it["y"], col, 7, tag)
    return out


def _counts(icons: dict) -> dict:
    c = {k: len(v) for k, v in icons.items()}
    for kind in ("hero", "yellow_icon", "red_dot", "red_ring", "blue_round",
                 "green_dot", "enemy_tower_edge_bottom"):
        c[kind] = sum(1 for it in icons["others"] if it["kind"] == kind)
    return c


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="小地图识别（离线，只读 PNG）")
    ap.add_argument("--selftest", nargs="+", metavar="PNG", help="对若干帧打印检测结果")
    ap.add_argument("--overlay", nargs=2, metavar=("IN", "OUT"), help="把检测结果画在原图上存盘")
    a = ap.parse_args(argv)
    if not a.selftest and not a.overlay:
        ap.print_help()
        return 2

    if a.selftest:
        n_hit = 0
        rows = []
        print(f"{'frame':16s} {'rect':26s} {'self':>16s} {'ally':>4s} {'enemy':>5s} "
              f"{'camp':>4s} {'hero':>4s} {'edge':>4s} {'yellow':>6s} {'others':>6s} {'terr':>5s}")
        for p in a.selftest:
            frame = read_frame(p)
            rect = minimap_rect(frame)
            self_m = find_self_marker(frame)
            icons = detect_icons(frame)
            c = _counts(icons)
            terr = content_terrain_fraction(frame)
            n_hit += 1 if self_m else 0
            rows.append((Path(p).name, rect, self_m, c))
            self_s = "None" if not self_m else f"({self_m['x']},{self_m['y']}) {self_m['conf']:.2f}"
            print(f"{Path(p).name:16s} {str(rect):26s} {self_s:>16s} "
                  f"{c['ally_towers']:4d} {c['enemy_towers']:5d} {c['camps']:4d} "
                  f"{c['hero']:4d} {c['enemy_tower_edge_bottom']:4d} "
                  f"{c['yellow_icon']:6d} {c['others']:6d} {terr:5.2f}")
        print(f"\n自家标记命中 {n_hit}/{len(a.selftest)} 帧")
        for i, name in enumerate(("x0", "y0", "x1", "y1")):
            vals = sorted(r[1][i] for r in rows)
            print(f"  rect.{name}: min={vals[0]} median={vals[len(vals)//2]} max={vals[-1]}")

    if a.overlay:
        src, dst = a.overlay
        frame = read_frame(src)
        rect = minimap_rect(frame)
        img = _draw(frame, rect, find_self_marker(frame), detect_icons(frame))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        write_frame(dst, img)
        print(f"overlay -> {dst}  ({img.shape[1]}x{img.shape[0]})  rect={rect}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))