
"""
dual_overlay.py
Mostra uma imagem duas vezes lado a lado:
- Esquerda: bbox/OBB predito por um modelo YOLO-OBB (Ultralytics).
- Direita: bbox(s) lidos de um TXT (formato YOLO: class cx cy w h normalizados).

Uso:
  python dual_overlay.py --image caminho/para/imagem.jpg \
                         --labels_txt gt_train_NOVO_17.5k.txt \
                         --model caminho/para/best.pt \
                         --stem_strategy name  # name | parent | auto
                         [--class_filter 2 86] \
                         [--save saida.png]

Requisitos:
  pip install ultralytics opencv-python numpy matplotlib
"""

from __future__ import annotations
import argparse
import os
import re
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Comparar bboxes: YOLO-OBB vs. TXT (YOLO xywhn).")
    ap.add_argument("--image", required=True, help="Caminho da imagem.")
    ap.add_argument("--labels_txt", required=True, help="Arquivo TXT com ground truth no formato especial enviado.")
    ap.add_argument("--model", default=None, help="Caminho do modelo YOLO-OBB (Ultralytics .pt). Se omitido, não desenha predição.")
    ap.add_argument("--stem_strategy", default="auto", choices=["auto","name","parent"],
                    help="Como inferir o 'stem' para buscar no TXT: "
                         "'name' usa o nome do arquivo sem extensão; "
                         "'parent' usa o nome da pasta; "
                         "'auto' tenta name e parent nesta ordem.")
    ap.add_argument("--class_filter", nargs="*", type=int, default=None,
                    help="Lista de ids de classe a manter (ex: 2 86). Se omitido, mantém todas.")
    ap.add_argument("--save", default=None, help="Se setado, salva a imagem de saída neste caminho.")
    return ap.parse_args()

# --------------------- Utilidades de desenho ---------------------

def draw_rectangles(img: np.ndarray, rects: List[Tuple[int,int,int,int]], color=(0,255,0), labels: Optional[List[str]]=None, thickness:int=2):
    out = img.copy()
    for i,(x1,y1,x2,y2) in enumerate(rects):
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            (tw, th), baseline = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(0, y1-5)
            cv2.rectangle(out, (x1, ty-th-baseline-2), (x1+tw+6, ty), color, -1)
            cv2.putText(out, labels[i], (x1+3, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return out

def draw_polygons(img: np.ndarray, polys: List[np.ndarray], color=(255,0,0), labels: Optional[List[str]]=None, thickness:int=2):
    out = img.copy()
    for i,poly in enumerate(polys):
        pts = poly.reshape(-1,1,2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            # coloca label no primeiro vértice
            x1,y1 = pts[0,0]
            (tw, th), baseline = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(0, y1-5)
            cv2.rectangle(out, (x1, ty-th-baseline-2), (x1+tw+6, ty), color, -1)
            cv2.putText(out, labels[i], (x1+3, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return out

# --------------------- Parsing do TXT (GT) ---------------------

def iter_blocks(lines: List[str]):
    """
    O arquivo fornecido contém blocos do tipo:
      === n01006.txt ===
      sfq9h56
      86 0.477083 0.630469 0.245833 0.048438
      2  0.472917 0.423438 0.837500 0.603125
    """
    current_header = None
    current_stem = None
    current_items = []
    sep_re = re.compile(r"^===\s+(.+?)\s+===\s*$")
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = sep_re.match(ln)
        if m:
            # flush
            if current_header is not None:
                yield (current_header, current_stem, current_items)
            current_header = m.group(1)
            current_stem = None
            current_items = []
        else:
            if current_stem is None and ln and not ln[0].isdigit():
                current_stem = ln.strip()
            else:
                parts = ln.split()
                if len(parts) == 5 and all(p.replace('.','',1).isdigit() for p in parts[1:]):
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    current_items.append((cls, cx, cy, w, h))
    # final flush
    if current_header is not None:
        yield (current_header, current_stem, current_items)

def get_gt_boxes_for_stem(labels_txt_path: str, stem: str, class_filter: Optional[List[int]], W: int, H: int):
    with open(labels_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rects = []
    labels = []
    for header, blk_stem, items in iter_blocks(lines):
        if blk_stem == stem:
            for cls, cx, cy, w, h in items:
                if (class_filter is None) or (cls in class_filter):
                    # converter de xywh normalizado para pixels e depois para x1y1x2y2
                    x1 = int((cx - w/2.0) * W)
                    y1 = int((cy - h/2.0) * H)
                    x2 = int((cx + w/2.0) * W)
                    y2 = int((cy + h/2.0) * H)
                    # clamp
                    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
                    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
                    rects.append((x1,y1,x2,y2))
                    labels.append(f"gt cls={cls}")
            break
    return rects, labels

# --------------------- Predição YOLO-OBB ---------------------

def predict_obb_polys(image_bgr: np.ndarray, model_path: Optional[str], class_filter: Optional[List[int]]):
    """
    Retorna lista de polígonos (4 pontos) em coordenadas de pixel e labels.
    Exige 'ultralytics' com suporte a OBB.
    """
    if model_path is None:
        return [], []
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[AVISO] ultralytics não disponível ou com erro:", e)
        return [], []

    model = YOLO(model_path)
    # Para OBB em Ultralytics, usar task='obb' quando apropriado
    try:
        res = model.predict(source=image_bgr[:, :, ::-1], verbose=False, task='obb')  # usa RGB
    except TypeError:
        # versões antigas podem ignorar 'task'; ainda assim retornam 'boxes' ou 'obb'
        res = model.predict(source=image_bgr[:, :, ::-1], verbose=False)

    if not res:
        return [], []

    r0 = res[0]
    polys = []
    labels = []

    # Tentar extrair OBB como 4 vértices (xyxyxyxy) se existir
    obb = getattr(r0, 'obb', None)
    boxes = getattr(r0, 'boxes', None)

    if obb is not None and hasattr(obb, 'xyxyxyxy'):
        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()  # (N,8)
        clses = obb.cls.cpu().numpy().astype(int) if hasattr(obb, 'cls') else np.zeros(len(xyxyxyxy), dtype=int)
        for poly8, c in zip(xyxyxyxy, clses):
            if (class_filter is None) or (int(c) in class_filter):
                poly = poly8.reshape(4,2)
                polys.append(poly)
                labels.append(f"pred cls={int(c)}")
    elif boxes is not None and hasattr(boxes, 'xyxy'):
        # fallback: caixas axis-aligned
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        clses = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.zeros(len(xyxy), dtype=int)
        for (x1,y1,x2,y2), c in zip(xyxy, clses):
            if (class_filter is None) or (int(c) in class_filter):
                polys.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))
                labels.append(f"pred cls={int(c)}")
    else:
        # nenhum resultado
        pass

    return polys, labels

# --------------------- Lógica principal ---------------------

def infer_stem(img_path: str, strategy: str) -> List[str]:
    name_stem = os.path.splitext(os.path.basename(img_path))[0]
    parent_stem = os.path.basename(os.path.dirname(img_path))
    if strategy == "name":
        return [name_stem]
    if strategy == "parent":
        return [parent_stem]
    # auto
    # algumas bases usam o código do arquivo sem extensão (ex: 'sgc2i73') como stem;
    # outras podem usar o nome da pasta.
    return [name_stem, parent_stem]

def main():
    args = parse_args()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {args.image}")
    H, W = img_bgr.shape[:2]

    # Predição YOLO-OBB (esquerda)
    pred_polys, pred_labels = predict_obb_polys(img_bgr, args.model, args.class_filter)
    left = draw_polygons(img_bgr, pred_polys, color=(255,0,0), labels=pred_labels, thickness=2)

    # Caso não haja predição ou modelo, apenas copia a imagem
    if len(pred_polys) == 0:
        left = img_bgr.copy()

    # GT do TXT (direita)
    stems = infer_stem(args.image, args.stem_strategy)
    gt_rects, gt_labels = [], []
    for st in stems:
        gt_rects, gt_labels = get_gt_boxes_for_stem(args.labels_txt, st, args.class_filter, W, H)
        if gt_rects:
            break
    right = draw_rectangles(img_bgr, gt_rects, color=(0,255,0), labels=gt_labels, thickness=2)

    # Compor lado a lado (BGR->RGB para plot)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    combo = np.hstack([left_rgb, right_rgb])

    # Mostrar com matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(combo)
    plt.axis('off')
    plt.title("Esq: YOLO-OBB | Dir: TXT (xywh normalizado)")
    plt.tight_layout()

    if args.save:
        # salvar com cv2 no espaço BGR
        combo_bgr = cv2.cvtColor(combo, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(args.save, combo_bgr)
        if not ok:
            print("[AVISO] Não foi possível salvar em", args.save)

    plt.show()

if __name__ == "__main__":
    main()
