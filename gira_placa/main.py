import cv2
import csv
import sys
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, List

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def four_point_warp(image: np.ndarray, src_pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

def angle_from_box(ordered_pts: np.ndarray) -> float:
    tl, tr, br, bl = ordered_pts
    v = tr - tl
    return float(np.degrees(np.arctan2(v[1], v[0])))

PLATE_ASPECT = 400.0/130.0
OUT_WIDTH = 384
OUT_HEIGHT = int(round(OUT_WIDTH/PLATE_ASPECT))

def detect_and_rectify(
    image_path: str,
    model: Optional[YOLO] = None,
    model_path: Optional[str] = None,
    class_filter: Optional[List[str]] = None,
    conf_thres: float = 0.1,
    iou_thres: float = 0.7,
    imgsz: int = 3000,
    save_dir: str = "outputs"
):
    """
    Processa UMA imagem. Se 'model' não for fornecido, carrega a partir de model_path.
    Retorna um dicionário com caminhos salvos e metadados da melhor detecção.
    """
    img = cv2.imread(str(image_path))
    assert img is not None, f"Não consegui ler {image_path}"

    if model is None:
        assert model_path is not None, "Forneça 'model' ou 'model_path'."
        model = YOLO(model_path)

    results = model(img, conf=conf_thres, iou=iou_thres, imgsz=imgsz)
    if not results:
        raise RuntimeError("Nenhum resultado retornado pelo modelo.")

    r = results[0]

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    vis_path = str(Path(save_dir) / f"{stem}_pred_vis.jpg")
    vis = r.plot()
    cv2.imwrite(vis_path, vis)

    names = r.names
    available = [names[i] for i in range(len(names))]

    polys = None
    confs = None
    clses = None
    angle_deg_model = None

    if getattr(r, "obb", None) is not None and r.obb is not None and len(r.obb) > 0:
        polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        confs = r.obb.conf.cpu().numpy().ravel()
        clses = r.obb.cls.cpu().numpy().astype(int)
        xywhr = r.obb.xywhr.cpu().numpy()  # cx, cy, w, h, rot(rad)
    else:
        if getattr(r, "boxes", None) is None or r.boxes is None or len(r.boxes) == 0:
            raise RuntimeError("Nenhuma detecção (OBB/boxes) encontrada. Veja a visualização.")
        b = r.boxes.xyxy.cpu().numpy()
        polys = np.stack([b[:, [0,1]], b[:, [2,1]], b[:, [2,3]], b[:, [0,3]]], axis=1)
        confs = r.boxes.conf.cpu().numpy().ravel()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        xywhr = None

    idxs = list(range(len(polys)))
    if class_filter:
        wanted = {c.lower() for c in class_filter}
        idxs_pref = [i for i in idxs if names[clses[i]].lower() in wanted]
        if idxs_pref:
            idxs = idxs_pref

    best_i = max(idxs, key=lambda i: confs[i])
    poly = polys[best_i]
    pts_ordered = order_points(poly)
    angle_deg_geom = angle_from_box(pts_ordered)
    if xywhr is not None:
        angle_deg_model = float(np.degrees(xywhr[best_i, 4]))

    warped = four_point_warp(img, pts_ordered, OUT_WIDTH, OUT_HEIGHT)
    out_img = str(Path(save_dir) / f"{stem}_plate_rectified.png")
    cv2.imwrite(out_img, warped)

    cv2.polylines(img, [pts_ordered.astype(int)], True, (0,255,0), 2)
    det_poly_path = str(Path(save_dir) / f"{stem}_det_poly.jpg")
    cv2.imwrite(det_poly_path, img)

    return {
        "source_image": str(image_path),
        "rectified_path": out_img,
        "pred_vis_path": vis_path,
        "det_poly_path": det_poly_path,
        "angle_deg_geom": angle_deg_geom,
        "angle_deg_model": angle_deg_model,
        "class_name": names[clses[best_i]],
        "confidence": float(confs[best_i]),
        "available_class_names": available
    }

def iter_images(input_dir: Path, recursive: bool = True):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    if recursive:
        for ext in exts:
            yield from input_dir.rglob(ext)
    else:
        for ext in exts:
            yield from input_dir.glob(ext)

def process_folder(
    input_dir: str,
    model_path: str,
    save_dir: str = "outputs",
    class_filter: Optional[List[str]] = None,
    conf_thres: float = 0.1,
    iou_thres: float = 0.7,
    imgsz: int = 3000,
    recursive: bool = True,
    make_summary_csv: bool = True
):
    """
    Processa TODAS as imagens em 'input_dir' (recursivo opcional) e salva resultados.
    Reutiliza o mesmo modelo YOLO para todas as imagens.
    """
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    rows = []
    total = 0
    ok = 0
    for img_path in iter_images(input_dir, recursive=recursive):
        total += 1
        try:
            info = detect_and_rectify(
                image_path=str(img_path),
                model=model,
                model_path=None,
                class_filter=class_filter,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                imgsz=imgsz,
                save_dir=str(save_dir)
            )
            ok += 1
            rows.append({
                "source_image": info["source_image"],
                "rectified_path": info["rectified_path"],
                "pred_vis_path": info["pred_vis_path"],
                "det_poly_path": info["det_poly_path"],
                "class_name": info["class_name"],
                "confidence": info["confidence"],
                "angle_deg_geom": info["angle_deg_geom"],
                "angle_deg_model": info["angle_deg_model"]
            })
            print(f"[OK] {img_path}")
        except Exception as e:
            rows.append({
                "source_image": str(img_path),
                "rectified_path": "",
                "pred_vis_path": "",
                "det_poly_path": "",
                "class_name": "",
                "confidence": "",
                "angle_deg_geom": "",
                "angle_deg_model": "",
                "error": str(e)
            })
            print(f"[ERRO] {img_path}: {e}")

    if make_summary_csv:
        csv_path = save_dir / "summary.csv"
        fieldnames = ["source_image","rectified_path","pred_vis_path","det_poly_path",
                      "class_name","confidence","angle_deg_geom","angle_deg_model","error"]
        # garante coluna 'error' mesmo quando não houve erro
        for r in rows:
            if "error" not in r:
                r["error"] = ""
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResumo salvo em: {csv_path}")

    print(f"\nConcluído: {ok}/{total} imagens processadas com sucesso.")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Processar todas as imagens de uma pasta com YOLO-OBB e retificar a placa.")
    p.add_argument("--input_dir", required=False, default=None, help="Pasta com imagens de origem.")
    p.add_argument("--image", required=False, default=None, help="(Opcional) Processar uma única imagem.")
    p.add_argument("--model_path", required=False, default=r"C:\OCR-PLACAS\gira_placa\yolo11n-obb.pt", help="Caminho do modelo YOLO (OBB).")
    p.add_argument("--save_dir", required=False, default="outputs", help="Diretório de saída.")
    p.add_argument("--class_filter", nargs="*", default=None, help='Filtro de classes. Ex: --class_filter "license-plate" "placa"')
    p.add_argument("--conf", type=float, default=0.1, help="Confiança mínima.")
    p.add_argument("--iou", type=float, default=0.7, help="IoU para NMS.")
    p.add_argument("--imgsz", type=int, default=3000, help="Tamanho da imagem de inferência.")
    p.add_argument("--no_recursive", action="store_true", help="Não varrer recursivamente.")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()

    if args.image:
        # modo de IMAGEM ÚNICA
        info = detect_and_rectify(
            image_path=args.image,
            model=None,
            model_path=args.model_path,
            class_filter=args.class_filter,
            conf_thres=args.conf,
            iou_thres=args.iou,
            imgsz=args.imgsz,
            save_dir=args.save_dir
        )
        print(info)
    elif args.input_dir:
        # modo de PASTA
        process_folder(
            input_dir=args.input_dir,
            model_path=args.model_path,
            save_dir=args.save_dir,
            class_filter=args.class_filter,
            conf_thres=args.conf,
            iou_thres=args.iou,
            imgsz=args.imgsz,
            recursive=not args.no_recursive,
            make_summary_csv=True
        )
    else:
        print("Informe --image OU --input_dir.")
        sys.exit(1)
