
import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ======= Config defaults (can be overridden by CLI) =======
PLATE_ASPECT = 400.0 / 130.0  # ~3.0769
OUT_WIDTH = 384
OUT_HEIGHT = int(round(OUT_WIDTH / PLATE_ASPECT))

# Heuristics for angle estimation from AABB-only
CANNY_LO = 50
CANNY_HI = 150
HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 0  # will be set relative to ROI width
HOUGH_MAX_LINE_GAP = 10
# weight to prefer near-horizontal lines typical of plates
HORIZONTAL_BIAS_DEG = 10.0  # degrees around 0/180


# ======= Geometry helpers =======
def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as [tl, tr, br, bl]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.vstack([tl, tr, br, bl]).astype(np.float32)


def four_point_warp(image: np.ndarray, src_pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


def aabb_to_quad_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> np.ndarray:
    """Convert normalized YOLO AABB (cx,cy,w,h in [0,1]) to 4-corner quad in pixel coords."""
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return order_points(quad)


def rbox_to_quad(cx: float, cy: float, w: float, h: float, theta_deg: float, W: int, H: int) -> np.ndarray:
    """Make a rotated rectangle quad from center, size (normalized) and angle (degrees)."""
    cxp, cyp = cx * W, cy * H
    wp, hp = w * W, h * H
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    # local corners (axis-aligned, centered)
    local = np.array([[-wp/2, -hp/2], [wp/2, -hp/2], [wp/2, hp/2], [-wp/2, hp/2]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = (local @ R.T) + np.array([cxp, cyp], dtype=np.float32)
    return order_points(rotated)


# ======= I/O helpers =======
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image(images_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    """Try a set of base names against common image extensions; return first that exists."""
    for base in candidates:
        if not base:
            continue
        base = base.strip()
        # if base has an extension already, test directly
        p = images_dir / base
        if p.suffix.lower() in IMG_EXTS and p.exists():
            return p
        # otherwise try common exts
        for ext in IMG_EXTS:
            q = images_dir / f"{base}{ext}"
            if q.exists():
                return q
    return None


# ======= Parsing ground-truth TXT =======
BLOCK_HEADER_RE = re.compile(r"^\s*===\s*([^.=\s]+)\.txt\s*===\s*$")


def parse_gt_blocks(gt_path: Path) -> Iterable[Dict]:
    """
    Parse a GT file with blocks of the form:

        === n00001.txt ===
        pph2i17
        86 0.318750 0.750000 0.229167 0.071875
        2  0.490625 0.501563 0.906250 0.659375

    Yields dicts with keys: 'block_id', 'second_id', 'entries' (list of dicts).
    Each entry has:
        - class_id (int)
        - nums (list of floats)  -> 4 nums (cx,cy,w,h) OR 8 nums (x1,y1,...,x4,y4) OR 5 nums (cx,cy,w,h,theta_deg)
    """
    with gt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    i = 0
    N = len(lines)
    while i < N:
        m = BLOCK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        block_id = m.group(1)
        i += 1
        # second line (often an image hash/base name)
        second_id = None
        if i < N and lines[i].strip() and not BLOCK_HEADER_RE.match(lines[i]):
            second_id = lines[i].strip()
            i += 1

        entries = []
        # collect until blank line or next header
        while i < N and lines[i].strip() and not BLOCK_HEADER_RE.match(lines[i]):
            parts = lines[i].strip().split()
            if parts and parts[0].lstrip("+-").isdigit():
                cls = int(parts[0])
                nums = [float(x) for x in parts[1:]]
                entries.append({"class_id": cls, "nums": nums})
            i += 1

        # skip blank line
        while i < N and not lines[i].strip():
            i += 1

        yield {"block_id": block_id, "second_id": second_id, "entries": entries}


def choose_plate_entry(entries: List[Dict], target_class: Optional[int]) -> Optional[Dict]:
    """Pick the plate bbox entry. Prefer target_class; else first with 4,5, or 8 numbers."""
    if target_class is not None:
        for e in entries:
            if e["class_id"] == int(target_class):
                return e
    # fallback: first plausible bbox
    for e in entries:
        if len(e["nums"]) in (4, 5, 8):
            return e
    return None


# ======= Angle estimation from AABB-only =======
def estimate_angle_from_roi(roi: np.ndarray) -> Optional[float]:
    """
    Estimate dominant near-horizontal angle (degrees) inside ROI using Canny + HoughLinesP.
    Returns angle in degrees where 0 means horizontal (in image x-axis), positive CCW.
    If no reliable angle is found, returns None.
    """
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI, L2gradient=True)

    h, w = edges.shape[:2]
    min_len = max(int(0.4 * w), 20)  # prefer long lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(HOUGH_THRESH, 20),
                            minLineLength=min_len, maxLineGap=HOUGH_MAX_LINE_GAP)
    if lines is None:
        return None

    # Gather angles in degrees
    angles = []
    weights = []
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = l.tolist()
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0 and dy == 0:
            continue
        length = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(dy, dx))  # [-180,180]
        # Normalize to [-90,90] to treat 0/180 the same
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180

        # Weight by length, and bias towards near-horizontal segments (typical of plates)
        horiz_bias = max(0.0, (HORIZONTAL_BIAS_DEG - min(abs(ang), abs(90 - abs(ang)))) / HORIZONTAL_BIAS_DEG)
        wgt = length * (1.0 + 0.5 * horiz_bias)
        angles.append(ang)
        weights.append(wgt)

    if not angles:
        return None

    # Weighted average robust to outliers
    ang = float(np.average(angles, weights=np.clip(weights, 1e-3, None)))
    return ang


def quad_from_aabb_with_angle(cx: float, cy: float, w: float, h: float,
                              theta_deg: float, W: int, H: int,
                              scale: float = 1.05) -> np.ndarray:
    """Build a rotated quad from normalized AABB center/size and an angle estimate."""
    cxp, cyp = cx * W, cy * H
    wp, hp = w * W * scale, h * H * scale
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    local = np.array([[-wp/2, -hp/2], [wp/2, -hp/2], [wp/2, hp/2], [-wp/2, hp/2]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = (local @ R.T) + np.array([cxp, cyp], dtype=np.float32)
    return order_points(rotated)


# ======= Main processing =======
def process(
    gt_file: Path,
    images_dir: Path,
    out_dir: Path,
    target_class: Optional[int],
    out_w: int,
    out_h: int,
    save_vis: bool = True,
    raise_on_missing_image: bool = False,
    fallback_minarearect: bool = True,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total, ok = 0, 0

    for blk in parse_gt_blocks(gt_file):
        total += 1
        block_id = blk["block_id"]
        second_id = blk["second_id"]
        e = choose_plate_entry(blk["entries"], target_class)

        # resolve image path (try block_id and second_id as base names)
        img_path = find_image(images_dir, [block_id, second_id])
        if img_path is None:
            msg = f"Imagem não encontrada para blocos {block_id!r} / {second_id!r}"
            if raise_on_missing_image:
                raise FileNotFoundError(msg)
            rows.append({"block_id": block_id, "image": "", "rectified": "", "error": msg})
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Falha ao ler {img_path}")
            H, W = img.shape[:2]

            if e is None:
                raise RuntimeError("Nenhuma entrada de placa encontrada no bloco.")

            nums = e["nums"]
            quad: Optional[np.ndarray] = None
            used_angle = None
            used_strategy = ""

            if len(nums) == 4:
                cx, cy, w, h = nums
                # First try: estimate angle within AABB via Hough
                # Build pixel ROI
                x1 = max(0, int(round((cx - w / 2.0) * W)))
                y1 = max(0, int(round((cy - h / 2.0) * H)))
                x2 = min(W - 1, int(round((cx + w / 2.0) * W)))
                y2 = min(H - 1, int(round((cy + h / 2.0) * H)))
                roi = img[y1:y2, x1:x2].copy()
                angle = estimate_angle_from_roi(roi)

                if angle is not None:
                    quad = quad_from_aabb_with_angle(cx, cy, w, h, angle, W, H, scale=1.08)
                    used_angle = angle
                    used_strategy = "Hough"
                elif fallback_minarearect:
                    # Fallback: threshold and minAreaRect to get a rotated box guess
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    # adaptive-ish threshold
                    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        # choose largest contour
                        cmax = max(cnts, key=cv2.contourArea)
                        rect = cv2.minAreaRect(cmax)  # ((cx,cy),(w,h),angle)
                        (rcx, rcy), (rw, rh), ang = rect
                        # convert local ROI rect to global quad
                        box = cv2.boxPoints(rect)  # 4x2
                        box[:, 0] += x1
                        box[:, 1] += y1
                        quad = order_points(box)
                        used_angle = ang
                        used_strategy = "minAreaRect"
                    else:
                        # final fallback: axis-aligned (no rotation info)
                        quad = aabb_to_quad_xyxy(cx, cy, w, h, W, H)
                        used_strategy = "AABB"
                else:
                    quad = aabb_to_quad_xyxy(cx, cy, w, h, W, H)
                    used_strategy = "AABB"

            elif len(nums) == 5:
                cx, cy, w, h, theta_deg = nums
                quad = rbox_to_quad(cx, cy, w, h, theta_deg, W, H)
                used_angle = theta_deg
                used_strategy = "RBOX"
            elif len(nums) == 8:
                pts = np.array(nums, dtype=np.float32).reshape(4, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                quad = order_points(pts)
                used_strategy = "QUAD8"
            else:
                raise RuntimeError(f"Formato de bbox não suportado: {len(nums)} valores")

            warped = four_point_warp(img, quad, out_w, out_h)

            stem = img_path.stem
            rect_path = out_dir / f"{stem}_rectified.png"
            cv2.imwrite(str(rect_path), warped)

            if save_vis:
                vis = img.copy()
                cv2.polylines(vis, [quad.astype(int)], True, (0, 255, 0), 2)
                # annotate strategy/angle
                txt = used_strategy
                if used_angle is not None:
                    txt += f" ({used_angle:.1f}°)"
                cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                vis_path = out_dir / f"{stem}_poly.jpg"
                cv2.imwrite(str(vis_path), vis)
            else:
                vis_path = ""

            ok += 1
            rows.append(
                {
                    "block_id": block_id,
                    "image": str(img_path),
                    "rectified": str(rect_path),
                    "vis": str(vis_path) if save_vis else "",
                    "class_id": e.get("class_id", ""),
                    "format_len": len(nums),
                    "strategy": used_strategy,
                    "angle_est": "" if used_angle is None else f"{used_angle:.3f}",
                }
            )

        except Exception as ex:
            rows.append({"block_id": block_id, "image": str(img_path), "rectified": "", "error": str(ex)})

    # write summary CSV
    csv_path = out_dir / "summary.csv"
    fieldnames = ["block_id", "image", "rectified", "vis", "class_id", "format_len", "strategy", "angle_est", "error"]
    for r in rows:
        if "error" not in r:
            r["error"] = ""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Resumo salvo em: {csv_path}")
    print(f"Concluído: {ok}/{total} placas retificadas.")
    return ok, total


def build_argparser():
    p = argparse.ArgumentParser(description="Retificar placas a partir de coordenadas em TXT (sem rodar YOLO).")
    p.add_argument("--gt_file", required=True, help="Caminho para o TXT de ground-truth (ex: gt_teste_NOVO_17.5k.txt).")
    p.add_argument("--images_dir", required=True, help="Pasta com as imagens originais.")
    p.add_argument("--out_dir", default="rectified_outputs", help="Pasta de saída.")
    p.add_argument("--target_class", type=int, default=86, help="Classe alvo da PLACA dentro do TXT (padrão: 86).")
    p.add_argument("--out_width", type=int, default=OUT_WIDTH, help="Largura da placa retificada.")
    p.add_argument("--out_height", type=int, default=OUT_HEIGHT, help="Altura da placa retificada.")
    p.add_argument("--no_vis", action="store_true", help="Não salvar visualização com polígono.")
    p.add_argument("--strict", action="store_true", help="Falhar se alguma imagem não for encontrada.")
    p.add_argument("--no_minarearect", action="store_true", help="Desativar fallback com minAreaRect para AABB.")
    return p


def main():
    args = build_argparser().parse_args()
    gt_file = Path(args.gt_file)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    save_vis = not args.no_vis
    process(
        gt_file=gt_file,
        images_dir=images_dir,
        out_dir=out_dir,
        target_class=args.target_class,
        out_w=args.out_width,
        out_h=args.out_height,
        save_vis=save_vis,
        raise_on_missing_image=args.strict,
        fallback_minarearect=not args.no_minarearect,
    )


if __name__ == "__main__":
    main()
