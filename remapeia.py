import os, glob

# aponte para suas pastas de labels
BASE = r"C:\OCR-PLACAS\yolo-obb-train\labels"
for split in ["train", "val"]:
    for p in glob.glob(os.path.join(BASE, split, "*.txt")):
        out_lines = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # força id de classe = 0
                parts[0] = "0"
                out_lines.append(" ".join(parts))
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
print("OK: rótulos remapeados para classe 0.")
