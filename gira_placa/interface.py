import os
import glob
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# UI
import tkinter as tk
from PIL import Image, ImageTk

# ===================== utilidades do seu main.py (mantidas/adaptadas) =====================

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def angle_from_box(ordered_pts: np.ndarray) -> float:
    tl, tr, br, bl = ordered_pts
    v = tr - tl
    return float(np.degrees(np.arctan2(v[1], v[0])))

def aabb_to_obb_four_points(cx, cy, w, h):
    # normalizado -> 4 cantos normalizados (x1,y1,x2,y1,x2,y2,x1,y2)
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def read_first_label_txt(txt_path: str) -> Optional[Tuple[int, List[float]]]:
    """
    Lê apenas a PRIMEIRA linha válida do arquivo de rótulo.
    Aceita:
      - OBB: 'cls x1 y1 x2 y2 x3 y3 x4 y4' (9 tokens)
      - AABB: 'cls cx cy w h' (5 tokens) -> converte para 4 cantos
    Retorna (cls, [x1,y1,...,x4,y4])  normalizado em [0,1]
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(parts[0])
            except:
                continue
            nums = list(map(float, parts[1:]))

            if len(nums) == 8:  # OBB direto
                return cls, nums
            elif len(nums) == 4:  # AABB -> OBB
                cx, cy, w, h = nums
                return cls, aabb_to_obb_four_points(cx, cy, w, h)
            else:
                # ignora linhas que não são 5 ou 9 tokens
                continue
    return None

def draw_poly_norm(img_bgr: np.ndarray, poly_norm: List[float], color=(0,0,255), thickness=2):
    h, w = img_bgr.shape[:2]
    pts = np.array([
        [poly_norm[0]*w, poly_norm[1]*h],
        [poly_norm[2]*w, poly_norm[3]*h],
        [poly_norm[4]*w, poly_norm[5]*h],
        [poly_norm[6]*w, poly_norm[7]*h],
    ], dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(img_bgr, [pts], True, color, thickness)
    return pts.reshape(-1,2).astype(np.float32)

# ===================== inferência com OBB (modelo) =====================

class OBBPredictor:
    def __init__(self, model_path: str, conf_thres=0.1, iou_thres=0.7, imgsz=1536, class_filter: Optional[List[str]]=None, device: str='cpu'):
        self.model = YOLO(model_path)
        self.conf = conf_thres
        self.iou = iou_thres
        self.imgsz = imgsz
        self.class_filter = {c.lower() for c in class_filter} if class_filter else None
        self.device = device
        

    def predict_best_poly(self, img_bgr: np.ndarray):
        # roda o modelo; retorna polígono (4x2), conf, classe, angle_geom, angle_model(opt)
        results = self.model(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device)
        if not results:
            return None

        r = results[0]
        names = r.names if getattr(r, "names", None) else self.model.names
        def name_for(i):
            try:
                if isinstance(names, dict):
                    return names.get(int(i), f"class{int(i)}")
                elif isinstance(names, (list, tuple)):
                    return names[int(i)] if 0 <= int(i) < len(names) else f"class{int(i)}"
            except Exception:
                pass
            return f"class{int(i)}"

        # pega OBB se existir; senão cai p/ boxes AABB
        if getattr(r, "obb", None) is not None and r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            confs = r.obb.conf.cpu().numpy().ravel()
            clses = r.obb.cls.cpu().numpy().astype(int)
            xywhr = r.obb.xywhr.cpu().numpy()  # [cx,cy,w,h,rot(rad)]
        else:
            if getattr(r, "boxes", None) is None or r.boxes is None or len(r.boxes) == 0:
                return None
            b = r.boxes.xyxy.cpu().numpy()
            polys = np.stack([b[:, [0,1]], b[:, [2,1]], b[:, [2,3]], b[:, [0,3]]], axis=1)
            confs = r.boxes.conf.cpu().numpy().ravel()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            xywhr = None

        idxs = list(range(len(polys)))
        if self.class_filter:
            idxs_f = [i for i in idxs if names[clses[i]].lower() in self.class_filter]
            if idxs_f:
                idxs = idxs_f

        best_i = max(idxs, key=lambda i: confs[i])
        poly = polys[best_i].astype(np.float32)
        poly = order_points(poly)
        angle_geom = angle_from_box(poly)
        angle_model = float(np.degrees(xywhr[best_i, 4])) if xywhr is not None else None

        return {
            "poly_xy": poly,
            "conf": float(confs[best_i]),
            "cls_name": name_for(clses[best_i]),
            "angle_geom": angle_geom,
            "angle_model": angle_model
        }

# ===================== UI (Tkinter) =====================

class OBBViewerApp:
    def __init__(
        self,
        root,
        images_dir: str,
        labels_dir: str,
        model_path: str,
        device: str = "cpu",
        class_names: Optional[Union[Dict[int, str], List[str]]] = None
    ):
        self.root = root
        self.root.title("OBB Viewer — modelo × label")

        # guarda caminhos
        self.images_dir = images_dir
        self.labels_dir = Path(labels_dir)

        # carrega lista de imagens
        import glob
        from pathlib import Path as _P
        self.images = sorted(sum([glob.glob(str(_P(images_dir) / f"*{ext}")) for ext in [".jpg", ".jpeg", ".png", ".bmp"]], []))
        assert self.images, f"Nenhuma imagem encontrada em {images_dir}"
        self.idx = 0

        # predictor (passe class_names ao criar)
        self.predictor = OBBPredictor(
            model_path=model_path,
            device=device,
            conf_thres=0.001,   # se estiver usando modelo zerado
            iou_thres=0.7,
            imgsz=1536,
            class_filter=None,
            class_names=class_names,   # <- aceita nomes customizados
        )

        # Carrega o modelo
        self.model = YOLO(model_path)  # YAML => random init; PT => pesos
        # Opcional: defina nomes para exibir algo amigável na UI
        if class_names is not None:
            # aceita dict {0:"placa"} ou lista ["placa"]
            self.model.names = class_names

    def _match_label_path(self, img_path: str) -> Optional[Path]:
        base = Path(img_path).stem
        p = self.labels_dir / f"{base}.txt"
        if p.exists():
            return p
        # fallback: se seus rótulos ficaram com 'obb' no nome
        p2 = self.labels_dir / f"{base}obb.txt"
        return p2 if p2.exists() else None

    def _cv_to_tk(self, img_bgr: np.ndarray, maxw=800) -> ImageTk.PhotoImage:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, maxw / float(w))
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(im)

    def show_current(self):
        img_path = self.images[self.idx]
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f"Falha ao abrir {img_path}"

        # ---- painel esquerdo: predição do modelo
        pred = self.predictor.predict_best_poly(img_bgr.copy())
        left_vis = img_bgr.copy()
        left_text = "Modelo: sem detecção"
        if pred:
            poly = pred["poly_xy"].astype(int).reshape(-1,1,2)
            cv2.polylines(left_vis, [poly], True, (0,255,0), 3)  # verde = modelo
            left_text = f"Modelo: {pred['cls_name']} | conf={pred['conf']:.3f} | ang_geom={pred['angle_geom']:.1f}°"
            if pred['angle_model'] is not None:
                left_text += f" | ang_model={pred['angle_model']:.1f}°"

        left_vis_disp = left_vis.copy()
        cv2.putText(left_vis_disp, left_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(left_vis_disp, left_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

        # ---- painel direito: anotação do TXT (1ª linha)
        right_vis = img_bgr.copy()
        label_path = self._match_label_path(img_path)
        right_text = "Label: arquivo não encontrado"
        if label_path and label_path.exists():
            parsed = read_first_label_txt(str(label_path))
            if parsed:
                cls, poly_norm = parsed
                pts = draw_poly_norm(right_vis, poly_norm, color=(0,0,255), thickness=3)  # vermelho = label
                ang = angle_from_box(order_points(pts))
                right_text = f"Label: cls={cls} | ang_geom={ang:.1f}° | {label_path.name}"
            else:
                right_text = f"Label: 1ª linha inválida | {label_path.name}"

        right_vis_disp = right_vis.copy()
        cv2.putText(right_vis_disp, right_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(right_vis_disp, right_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

        # converte para Tk e exibe
        self.left_imgtk  = self._cv_to_tk(left_vis_disp)
        self.right_imgtk = self._cv_to_tk(right_vis_disp)
        self.left_panel.configure(image=self.left_imgtk)
        self.right_panel.configure(image=self.right_imgtk)

        self.status.set(f"[{self.idx+1}/{len(self.images)}] {Path(img_path).name}")

    def next_image(self):
        self.idx = (self.idx + 1) % len(self.images)
        self.show_current()

    def prev_image(self):
        self.idx = (self.idx - 1) % len(self.images)
        self.show_current()

    def reload(self):
        self.show_current()

# ===================== entrypoint =====================

def launch_ui(images_dir, labels_dir, model_path, device="cpu"):
    root = tk.Tk()
    app = OBBViewerApp(
        root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        model_path=model_path,
        device=device,
        class_names={0: "placa"},   # opcional, útil p/ modelo zerado
    )
    root.mainloop()

if __name__ == "__main__":
    # AJUSTE estes caminhos:
    launch_ui(
        images_dir=r"C:\OCR-PLACAS\yolo-obb-train\images\train",
        labels_dir=r"C:\OCR-PLACAS\yolo-obb-train\labels\train",
        model_path=r"C:\OCR-PLACAS\yolo-obb-train\gira_placa\yolo11n-obb.pt",
        device="cpu"
    )

