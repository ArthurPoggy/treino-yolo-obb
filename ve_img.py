# yolo11_obb_detect.py
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ============ PARÂMETROS ============
image_path = r"C:\OCR-PLACAS\yolo-obb-train\images\train\n01010.jpg"         # imagem de origem
model_path = r"yolo11n-obb.pt"                         # pode ser seu best.pt obb treinado
out_image  = r"C:\caminho\saida\det_obb.jpg"           # onde salvar a imagem com OBB desenhado
out_txt    = r"C:\caminho\saida\det_obb.txt"           # onde salvar os vértices normalizados
conf_thres = 0.25
iou_thres  = 0.50
# ====================================

# Carrega imagem
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Não consegui abrir a imagem em: {image_path}")
h, w = img.shape[:2]

# Carrega modelo YOLO11-OBB
model = YOLO(model_path)

# Faz a predição (retorna 1 ou mais resultados por imagem)
results = model.predict(
    source=image_path,
    conf=conf_thres,
    iou=iou_thres,
    verbose=False
)

if not results or len(results) == 0:
    raise RuntimeError("Nenhum resultado retornado pelo modelo.")

r = results[0]

# Se quiser ver rapidamente a visualização automática do Ultralytics:
# annotated = r.plot()  # já vem com OBB desenhado
# cv2.imshow("YOLO11-OBB", annotated); cv2.waitKey(0)

# Vamos extrair as OBBs e desenhar manualmente para também exportar as coordenadas
overlay = img.copy()

# Verifica se o resultado tem OBB
if not hasattr(r, "obb") or r.obb is None:
    raise RuntimeError("O resultado não possui OBB (verifique se o modelo é OBB).")

# Pega os polígonos como 4 vértices (xyxyxyxy) e também classe/conf
# xyxyxyxy -> array (N, 8): [x1,y1, x2,y2, x3,y3, x4,y4]
polys = r.obb.xyxyxyxy  # tensor
cls_ids = r.boxes.cls if hasattr(r, "boxes") and r.boxes is not None else None
confs   = r.boxes.conf if hasattr(r, "boxes") and r.boxes is not None else None

if polys is None or len(polys) == 0:
    raise RuntimeError("O modelo não detectou nenhuma OBB nesta imagem.")

polys = polys.cpu().numpy().astype(int)
names = r.names if hasattr(r, "names") else {}

# Para exportar como YOLO-OBB normalizado (classe + 8 floats em [0,1])
def to_norm(x, y, W, H):
    return float(x)/float(W), float(y)/float(H)

export_lines = []
for i, p in enumerate(polys):
    # p = [x1,y1,x2,y2,x3,y3,x4,y4]
    pts = p.reshape(4, 2)

    # Desenha polígono na imagem
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Rótulo (classe + confiança) opcional
    if cls_ids is not None and confs is not None:
        cls_id = int(cls_ids[i])
        conf   = float(confs[i])
        label  = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
    else:
        cls_id = 0
        label  = f"id0"

    # Posiciona texto perto do primeiro vértice
    x0, y0 = int(pts[0,0]), int(pts[0,1])
    cv2.putText(overlay, label, (x0, max(0, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    # Exporta formato YOLO-OBB normalizado: class x1 y1 x2 y2 x3 y3 x4 y4
    norm_coords = []
    for (xx, yy) in pts:
        xn, yn = to_norm(xx, yy, w, h)
        norm_coords.extend([f"{xn:.6f}", f"{yn:.6f}"])
    line = f"{cls_id} " + " ".join(norm_coords)
    export_lines.append(line)

# Salva a imagem com OBB desenhado
Path(out_image).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(out_image, overlay)

# Mostra na tela
cv2.imshow("YOLO11-OBB detections", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salva TXT com os vértices normalizados
Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
with open(out_txt, "w", encoding="utf-8") as f:
    for line in export_lines:
        f.write(line + "\n")

print(f"[OK] OBBs desenhadas em: {out_image}")
print(f"[OK] Vértices normalizados em: {out_txt}")
