import argparse
import os
import re
import cv2
import numpy as np

def clamp01(v): 
    return max(0.0, min(1.0, float(v)))

def aabb_to_obb_four_points(cx, cy, w, h):
    """
    Converte (cx,cy,w,h) normalizados -> 4 cantos normalizados (sentido horário):
    (x1,y1) top-left, (x2,y1) top-right, (x2,y2) bottom-right, (x1,y2) bottom-left
    """
    x1, y1 = clamp01(cx - w/2), clamp01(cy - h/2)
    x2, y2 = clamp01(cx + w/2), clamp01(cy + h/2)
    return [x1, y1,  x2, y1,  x2, y2,  x1, y2]

def load_first_label(txt_path):
    """
    Lê apenas a PRIMEIRA linha válida no formato: 'cls cx cy w h' (normalizado).
    Retorna lista com UM item: [(cls, cx, cy, w, h)] ou [] se não achar.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except:
                continue
            # achou a 1ª válida -> retorna só ela
            return [(cls, cx, cy, w, h)]
    return []

def save_obb_labels(txt_out_path, obb_labels):
    """
    Salva rótulos OBB (4 cantos) normalizados: 'cls x1 y1 x2 y2 x3 y3 x4 y4'
    """
    os.makedirs(os.path.dirname(txt_out_path), exist_ok=True)
    with open(txt_out_path, "w", encoding="utf-8") as f:
        for cls, pts in obb_labels:
            f.write(f"{cls} " + " ".join(f"{p:.6f}" for p in pts) + "\n")

def draw_obb_on_image(img, polys_norm, thickness=2):
    """
    Desenha polígonos OBB (em normalizado) na imagem (em pixel).
    A 1ª anotação é vermelha, a 2ª é verde; as demais ficam azuis.
    """
    h, w = img.shape[:2]
    for i, poly in enumerate(polys_norm):
        # Cores em BGR (OpenCV)
        if i == 0:
            color = (0, 0, 255)   # vermelho
        elif i == 1:
            color = (0, 255, 0)   # verde
        else:
            color = (255, 0, 0)   # azul (opcional para as demais)

        pts = np.array([
            [poly[0]*w, poly[1]*h],
            [poly[2]*w, poly[3]*h],
            [poly[4]*w, poly[5]*h],
            [poly[6]*w, poly[7]*h],
        ], dtype=np.int32).reshape(-1, 1, 2)

        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def convert_single_image(image_path, labels_path, out_labels_path=None, out_image_path=None, show=False):
    # 1) carrega rótulos AABB
    items = load_first_label(labels_path)
    if not items:
        raise ValueError(f"Nenhuma linha válida em {labels_path}.")

    # 2) converte para OBB (4 cantos normalizados)
    obb_labels = []
    obb_polys = []
    for cls, cx, cy, w, h in items:
        pts = aabb_to_obb_four_points(cx, cy, w, h)
        obb_labels.append((cls, pts))
        obb_polys.append(pts)

    # 3) salva rótulos OBB
    if out_labels_path is None:
        root, _ = os.path.splitext(labels_path)
        out_labels_path = root + "_obb.txt"
    save_obb_labels(out_labels_path, obb_labels)

    # 4) desenha na imagem original (para conferir)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {image_path}")
    draw_obb_on_image(img, obb_polys, thickness=2)

    # 5) salva/mostra
    if out_image_path is None:
        root, ext = os.path.splitext(image_path)
        out_image_path = root + "_obb_preview" + ext
    cv2.imwrite(out_image_path, img)
    if show:
        cv2.imshow("OBB Preview", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[OK] Convertido OBB: {out_labels_path}")
    print(f"[OK] Preview salvo: {out_image_path}")

def split_mega_txt(mega_txt, out_dir):
    """
    Opcional: divide um 'arquivo grandão' com blocos do tipo:
    === n01006.txt ===
    <linhas 'cls cx cy w h'...>
    Retorna caminhos criados.
    """
    os.makedirs(out_dir, exist_ok=True)
    created = []
    current = None
    buf = []

    def flush():
        nonlocal current, buf
        if current and buf:
            dst = os.path.join(out_dir, current)
            with open(dst, "w", encoding="utf-8") as f:
                f.write("\n".join(buf) + "\n")
            created.append(dst)
        current, buf = None, []

    with open(mega_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = re.match(r"^===\s*(.+?)\s*===\s*$", line)
            if m:
                flush()
                name = m.group(1)
                current = name if name.endswith(".txt") else name + ".txt"
                continue
            if not line:
                continue
            # tenta formato 'cls cx cy w h'
            parts = line.split()
            if len(parts) == 5 and parts[0].isdigit():
                buf.append(line)
    flush()
    return created

def main():
    ap = argparse.ArgumentParser(description="AABB (cx,cy,w,h) -> YOLO-OBB (4 cantos) + visualização")
    ap.add_argument("--image", help="Caminho da imagem (ex.: images/n01006.jpg)")
    ap.add_argument("--labels", help="Caminho do .txt com linhas 'cls cx cy w h'")
    ap.add_argument("--out-labels", help="Saída do .txt OBB (opcional)")
    ap.add_argument("--out-image", help="Saída da imagem preview (opcional)")
    ap.add_argument("--show", action="store_true", help="Mostrar janela com preview")
    # modo opcional p/ arquivo grandão
    ap.add_argument("--mega-labels", help="Arquivo grandão com blocos '=== nome.txt ==='")
    ap.add_argument("--split-out", default="labels_split", help="Pasta de saída ao dividir mega .txt")
    args = ap.parse_args()

    if args.mega_labels:
        made = split_mega_txt(args.mega_labels, args.split_out)
        print(f"[OK] {len(made)} arquivos de rótulos criados em: {args.split_out}")
        return

    if not args.image or not args.labels:
        ap.error("Use --image e --labels (ou use --mega-labels para dividir o arquivo grandão).")

    convert_single_image(
        image_path=args.image,
        labels_path=args.labels,
        out_labels_path=args.out_labels,
        out_image_path=args.out_image,
        show=args.show
    )

if __name__ == "__main__":
    main()
