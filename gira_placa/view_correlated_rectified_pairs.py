#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter viewer for correlated plate rectification images.

- Far left: images like n00001.* (no suffix)  [SOURCE]
- Middle: images like n00001_rectified.*      [RECTIFIED]
- Right: images like n00001_plate_rectified.* [PLATE_RECTIFIED]

Match is by the common ID at the start (e.g., "n00001").

Controls
--------
- ← / → : previous / next pair
- Home / End: first / last pair
- G : go to specific ID (e.g., n00037)
- O : choose folders (left=source, middle=_rectified, right=_plate_rectified)
- S : toggle "fit" vs "actual size"
- Ctrl+S: save current triptych as a side-by-side composite PNG
- Esc / Q: quit

Usage (optional args):
----------------------
python view_correlated_rectified_pairs.py "C:\path\to\source" "C:\path\to\rectified" "C:\path\to\plate_rectified"

Requirements:
- Python 3.x
- Pillow (pip install pillow)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Accept IDs that start at the beginning, up to the first underscore or dot
ID_PATTERN = re.compile(r"^(?P<id>[^_.]+)")

def extract_id(filename: str) -> Optional[str]:
    m = ID_PATTERN.match(filename)
    return m.group("id") if m else None

def scan_triples(dir_source: Path, dir_rectified: Path, dir_plate_rectified: Path) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    """Return a sorted list of (id, src_path, rectified_path, plate_rectified_path)."""
    src_map: Dict[str, Path] = {}
    left_map: Dict[str, Path] = {}
    right_map: Dict[str, Path] = {}

    if dir_source and dir_source.exists():
        for p in dir_source.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                _id = extract_id(p.name)
                if _id:
                    src_map[_id] = p

    if dir_rectified and dir_rectified.exists():
        for p in dir_rectified.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_rectified" in p.stem:
                _id = extract_id(p.name)
                if _id:
                    left_map[_id] = p

    if dir_plate_rectified and dir_plate_rectified.exists():
        for p in dir_plate_rectified.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_plate_rectified" in p.stem:
                _id = extract_id(p.name)
                if _id:
                    right_map[_id] = p

    all_ids = sorted(set(src_map.keys()) | set(left_map.keys()) | set(right_map.keys()),
                     key=lambda s: (extract_numeric(s), s))
    triples = [(i, src_map.get(i), left_map.get(i), right_map.get(i)) for i in all_ids]
    return triples

def extract_numeric(s: str) -> int:
    # Extract number from IDs like "n00001"; fallback to 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0

def pil_load(path: Optional[Path]) -> Optional[Image.Image]:
    if not path or not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

class TripleViewer(tk.Tk):
    def __init__(self, dir_source: Optional[Path]=None, dir_left: Optional[Path]=None, dir_right: Optional[Path]=None):
        super().__init__()
        self.title("Triple Rectified Viewer")
        self.geometry("1600x800")
        self.minsize(1024, 560)

        # State
        self.dir_source = dir_source
        self.dir_left = dir_left
        self.dir_right = dir_right
        self.triples: List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]] = []
        self.idx = 0
        self.fit_to_window = True
        self.tk_src_img = None
        self.tk_left_img = None
        self.tk_right_img = None

        # UI
        self._build_ui()
        self._bind_keys()

        # Initial load
        self.reload_triples(initial=True)

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        self.lbl_dirs = ttk.Label(top, text=self._dirs_label(), wraplength=1200)
        self.lbl_dirs.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(top, text="Escolher Pastas (O)", command=self.choose_dirs).pack(side=tk.RIGHT)
        ttk.Button(top, text="Salvar Composto (Ctrl+S)", command=self.save_composite).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Ajuste: Janela (S)", command=self.toggle_fit).pack(side=tk.RIGHT, padx=4)

        # Middle: images
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_src = tk.Canvas(mid, bg="#1f1f1f", highlightthickness=1, highlightbackground="#444")
        self.canvas_left = tk.Canvas(mid, bg="#202124", highlightthickness=1, highlightbackground="#444")
        self.canvas_right = tk.Canvas(mid, bg="#202124", highlightthickness=1, highlightbackground="#444")

        self.canvas_src.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom bar
        bottom = ttk.Frame(self, padding=6)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = ttk.Label(bottom, text="0/0")
        self.lbl_status.pack(side=tk.LEFT)

        ttk.Button(bottom, text="⟨ Anterior (←)", command=self.prev_item).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="Próximo (→) ⟩", command=self.next_item).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="Ir para ID (G)", command=self.goto_id).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="Primeiro (Home)", command=self.first_item).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="Último (End)", command=self.last_item).pack(side=tk.RIGHT, padx=4)

        # Resize handling
        self.canvas_src.bind("<Configure>", lambda e: self.render_current())
        self.canvas_left.bind("<Configure>", lambda e: self.render_current())
        self.canvas_right.bind("<Configure>", lambda e: self.render_current())

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.prev_item())
        self.bind("<Right>", lambda e: self.next_item())
        self.bind("<Home>", lambda e: self.first_item())
        self.bind("<End>", lambda e: self.last_item())
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("q", lambda e: self.destroy())
        self.bind("Q", lambda e: self.destroy())
        self.bind("o", lambda e: self.choose_dirs())
        self.bind("O", lambda e: self.choose_dirs())
        self.bind("g", lambda e: self.goto_id())
        self.bind("G", lambda e: self.goto_id())
        self.bind("s", lambda e: self.toggle_fit())
        self.bind("S", lambda e: self.toggle_fit())
        self.bind("<Control-s>", lambda e: self.save_composite())

    def _dirs_label(self) -> str:
        s = str(self.dir_source) if self.dir_source else "(pasta SOURCE: n00001.*)"
        l = str(self.dir_left) if self.dir_left else "(pasta _rectified)"
        r = str(self.dir_right) if self.dir_right else "(pasta _plate_rectified)"
        return f"Fonte: {s}  |  Esquerda: {l}  |  Direita: {r}"

    def choose_dirs(self):
        new_src = filedialog.askdirectory(title="Escolha a pasta com IDs (ex.: n00001.*)")
        if not new_src:
            return
        new_left = filedialog.askdirectory(title="Escolha a pasta com *_rectified")
        if not new_left:
            return
        new_right = filedialog.askdirectory(title="Escolha a pasta com *_plate_rectified")
        if not new_right:
            return
        self.dir_source = Path(new_src)
        self.dir_left = Path(new_left)
        self.dir_right = Path(new_right)
        self.lbl_dirs.config(text=self._dirs_label())
        self.reload_triples()

    def reload_triples(self, initial: bool=False):
        if initial and (self.dir_source is None and self.dir_left is None and self.dir_right is None):
            # Try to auto-fill common Windows paths if they exist
            default_src = Path(r"C:\OCR-PLACAS\treino_carro_final_26_06\carros_17.5k_teste\carros_17.5k_teste\imgs\processed")
            default_left = Path(r"C:\OCR-PLACAS\yolo-obb-train\gira_placa\outputs")
            if default_src.exists():
                self.dir_source = default_src
            if default_left.exists():
                self.dir_left = default_left
            # Right dir (plate_rectified) must be chosen by the user unless guessed path is added here

        self.triples = scan_triples(self.dir_source or Path(),
                                    self.dir_left or Path(),
                                    self.dir_right or Path())
        self.idx = 0
        self.update_status()
        self.render_current()

    def current_item(self) -> Optional[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
        if not self.triples:
            return None
        return self.triples[self.idx]

    def update_status(self):
        n = len(self.triples)
        cur = self.idx + 1 if n else 0
        text = f"{cur}/{n}"
        ci = self.current_item()
        if ci:
            cid, sp, lp, rp = ci
            text += f"    ID: {cid}"
            text += f"    SRC:{sp.name if sp else '—'}    L:{lp.name if lp else '—'}    R:{rp.name if rp else '—'}"
        self.lbl_status.config(text=text)

    def render_current(self):
        ci = self.current_item()
        for canv in (self.canvas_src, self.canvas_left, self.canvas_right):
            canv.delete("all")

        if not ci:
            for canv in (self.canvas_src, self.canvas_left, self.canvas_right):
                canv.create_text(canv.winfo_width()/2, canv.winfo_height()/2,
                                 text="Nenhuma imagem encontrada.\nUse 'O' para escolher pastas.",
                                 fill="#ddd")
            return

        cid, sp, lp, rp = ci
        simg = pil_load(sp)
        limg = pil_load(lp)
        rimg = pil_load(rp)

        self._draw_on_canvas(self.canvas_src, simg, f"{cid}  (source)")
        self._draw_on_canvas(self.canvas_left, limg, f"{cid}  (rectified)")
        self._draw_on_canvas(self.canvas_right, rimg, f"{cid}  (plate_rectified)")
        self.update_status()

    def _draw_on_canvas(self, canv: tk.Canvas, img: Optional[Image.Image], caption: str):
        W = max(canv.winfo_width(), 2)
        H = max(canv.winfo_height(), 2)

        if img is None:
            canv.create_text(W/2, H/2, text="(imagem não encontrada)", fill="#ccc")
            canv.create_text(8, 8, text=caption, fill="#aaa", anchor="nw")
            return

        if self.fit_to_window:
            iw, ih = img.size
            scale = min(W / iw, H / ih)
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            disp = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            disp = img

        tkimg = ImageTk.PhotoImage(disp)
        # Keep reference
        if canv is self.canvas_src:
            self.tk_src_img = tkimg
        elif canv is self.canvas_left:
            self.tk_left_img = tkimg
        else:
            self.tk_right_img = tkimg

        canv.create_image(W//2, H//2, image=tkimg, anchor="center")
        canv.create_text(8, 8, text=caption, fill="#ddd", anchor="nw")

    # Navigation
    def prev_item(self):
        if not self.triples:
            return
        self.idx = (self.idx - 1) % len(self.triples)
        self.render_current()

    def next_item(self):
        if not self.triples:
            return
        self.idx = (self.idx + 1) % len(self.triples)
        self.render_current()

    def first_item(self):
        if not self.triples:
            return
        self.idx = 0
        self.render_current()

    def last_item(self):
        if not self.triples:
            return
        self.idx = len(self.triples) - 1
        self.render_current()

    def goto_id(self):
        if not self.triples:
            return
        ids = [cid for (cid, _, _, _) in self.triples]
        answer = simpledialog.askstring("Ir para ID", "Digite o ID (ex.: n00037):", parent=self)
        if not answer:
            return
        try_id = answer.strip()
        if try_id in ids:
            self.idx = ids.index(try_id)
            self.render_current()
        else:
            # try pad number portion
            num = re.findall(r"\d+", try_id)
            if num:
                num = num[0]
                candidates = [cid for cid in ids if re.search(rf"\b0*{num}\b", cid)]
                if candidates:
                    self.idx = ids.index(candidates[0])
                    self.render_current()
                    return
            messagebox.showinfo("Não encontrado", f"ID '{try_id}' não consta na lista.")

    def toggle_fit(self):
        self.fit_to_window = not self.fit_to_window
        self.render_current()

    def save_composite(self):
        ci = self.current_item()
        if not ci:
            return
        cid, sp, lp, rp = ci
        src = pil_load(sp)
        left = pil_load(lp)
        right = pil_load(rp)
        if src is None and left is None and right is None:
            messagebox.showwarning("Aviso", "Não há imagens para salvar.")
            return

        # Replace missing with dark boxes matching available height
        imgs = [im for im in (src, left, right) if im is not None]
        if not imgs:
            return

        # Make heights equal for a side-by-side composite of 3 columns
        target_h = max(im.height for im in imgs)
        def scale_to_h(im, H):
            if im is None:
                return Image.new("RGB", (int(H*1.5), H), (30, 30, 30))
            w = int(round(im.width * (H / im.height)))
            return im.resize((w, H), Image.LANCZOS)

        src_r  = scale_to_h(src,  target_h)
        left_r = scale_to_h(left, target_h)
        right_r= scale_to_h(right,target_h)

        comp_w = src_r.width + left_r.width + right_r.width
        comp = Image.new("RGB", (comp_w, target_h), (20, 20, 20))
        x = 0
        for im in (src_r, left_r, right_r):
            comp.paste(im, (x, 0))
            x += im.width

        # Save dialog
        fname = filedialog.asksaveasfilename(
            title="Salvar composto PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"{cid}_triptych.png"
        )
        if not fname:
            return
        comp.save(fname)
        messagebox.showinfo("Salvo", f"Composto salvo em:\n{fname}")

def main():
    import sys
    dir_source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    dir_left   = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    dir_right  = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    app = TripleViewer(dir_source, dir_left, dir_right)
    app.mainloop()

if __name__ == "__main__":
    main()
