# mega_to_obb_first.py
# Lê um mega .txt com blocos:
# === n01006.txt ===
# <linhas diversas...>
# <primeira linha "cls cx cy w h" (normalizado)>
# <demais linhas...>
#
# Para cada bloco, pega APENAS a 1ª linha válida "cls cx cy w h",
# converte para OBB (4 cantos) e salva em OUT_DIR/<nomebase>obb.txt
# Ex.: "=== n01006.txt ===" -> "n01006obb.txt"
#
# Uso:
#   python mega_to_obb_first.py --mega gt_teste_NOVO_17.5k.txt --out obb_labels
#
# Requisitos:
#   - Python 3.8+
#   - sem dependências externas

import os
import re
import argparse

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def aabb_to_obb_four_points(cx: float, cy: float, w: float, h: float):
    """
    Converte (cx,cy,w,h) normalizados -> 4 cantos normalizados (sentido horário):
    (x1,y1)=top-left, (x2,y1)=top-right, (x2,y2)=bottom-right, (x1,y2)=bottom-left
    """
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
    # formato alvo YOLO-OBB: x1 y1 x2 y1 x2 y2 x1 y2
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def write_obb_line(out_dir: str, header_name: str, cls_id: int, obb_pts):
    """
    Salva uma única linha: 'cls x1 y1 x2 y2 x3 y3 x4 y4'
    Nome do arquivo: <base>obb.txt  (ex.: n01006obb.txt)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = header_name
    # header pode vir como 'n01006.txt' — removemos a extensão se houver
    if base.lower().endswith(".txt"):
        base = base[:-4]
    out_name = f"{base}.txt"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} " + " ".join(f"{p:.6f}" for p in obb_pts) + "\n")
    return out_path

def process_mega(mega_path: str, out_dir: str):
    """
    Percorre o mega arquivo.
    Para cada bloco (=== nome.txt ===), captura a 1ª linha 'cls cx cy w h',
    converte e grava <nome>obb.txt. Ignora linhas não conformes (ex.: 'sfq9h56').
    """
    header_re = re.compile(r"^===\s*(.+?)\s*===\s*$")
    current_header = None
    taken_for_current = False  # já pegamos a 1ª linha válida deste bloco?

    total_blocks = 0
    total_written = 0

    with open(mega_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # novo bloco?
            m = header_re.match(line)
            if m:
                # inicia novo bloco
                current_header = m.group(1)
                taken_for_current = False
                total_blocks += 1
                continue

            if current_header is None:
                # ainda não entrou em um bloco válido
                continue

            if taken_for_current:
                # já escrevemos a 1ª linha desse bloco; ignorar demais
                continue

            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                # ignora linhas como 'sfq9h56', etc.
                continue

            # tenta parsear a 1ª linha válida
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except Exception:
                continue

            # converte para OBB 4 cantos
            obb = aabb_to_obb_four_points(cx, cy, w, h)

            # salva arquivo <base>obb.txt
            out_path = write_obb_line(out_dir, current_header, cls, obb)
            total_written += 1
            taken_for_current = True  # trava para este bloco

    print(f"[Resumo] Blocos encontrados: {total_blocks} | Arquivos gerados: {total_written}")
    print(f"[Saída] Pasta: {os.path.abspath(out_dir)}")

def main():
    ap = argparse.ArgumentParser(description="Extrai a 1ª linha de bbox de cada bloco e converte para YOLO-OBB (4 cantos).")
    ap.add_argument("--mega", required=True, help="Caminho do mega arquivo (ex.: gt_teste_NOVO_17.5k.txt)")
    ap.add_argument("--out", default="obb_labels", help="Pasta de saída dos arquivos <base>obb.txt")
    args = ap.parse_args()

    process_mega(args.mega, args.out)

if __name__ == "__main__":
    main()