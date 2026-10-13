#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Video Band Kirpici - MP4 ust/alt kirpma (otomatik + elle)

import os
import queue
import shutil
import base64
import tempfile
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
    import cv2
except ImportError as exc:
    print("Eksik paket:", exc)
    print("Kurulum: pip install numpy opencv-python imageio-ffmpeg")
    raise SystemExit(1)


MODES = [
    ("black", "Otomatik - siyah/beyaz bant"),
    ("motion", "Otomatik - hareketsiz bolge"),
    ("mixed", "Otomatik - karma (en fazla kirpar)"),
    ("manual", "Elle - cizgileri surukle"),
]


def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def open_cap(path):
    # Turce karakterli yollarda olmazsa gecici ASCII kopya dener
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, None
    cap.release()
    tmp = os.path.join(tempfile.gettempdir(), "vc_tmp_%d.mp4" % os.getpid())
    shutil.copy2(path, tmp)
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        rm(tmp)
        raise RuntimeError("Video acilamadi: " + os.path.basename(path))
    return cap, tmp


def rm(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def analyze(path, mode, sens):
    # satir satir "icerik" analizi -> (y0, y1, W, H)
    cap, tmp = open_cap(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        idxs = np.unique(np.linspace(0, total - 1, 20).round().astype(int))
    else:
        idxs = list(range(0, 200, 10))

    tex = np.zeros(H, np.float32)
    mot = np.zeros(H, np.float32)
    prev = None
    used = 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if (not ok) or (fr is None):
            continue
        if fr.shape[0] != H or fr.shape[1] != W:
            fr = cv2.resize(fr, (W, H))
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        std = g.std(axis=1)
        mean = g.mean(axis=1)
        rows = (std > 6.0) | ((mean > 30.0) & (mean < 225.0))
        tex += rows.astype(np.float32)
        if prev is not None:
            mot += np.abs(g - prev).mean(axis=1)
        prev = g
        used += 1

    cap.release()
    rm(tmp)

    if used == 0:
        raise RuntimeError("Kare okunamadi")

    ts = tex / float(used)
    mm = float(mot.max())
    if mm < 1.0:
        ms = np.zeros(H, np.float32)
    else:
        ms = mot / mm

    s = max(0.0, min(1.0, float(sens) / 100.0))
    a = ts > (0.05 + 0.50 * s)   # dokulu / normal parlaklik
    b = ms > (0.01 + 0.25 * s)   # zamanla degisen (hareketli)

    if mode == "black":
        mask = a
    elif mode == "motion":
        mask = b
    else:
        mask = np.logical_or(a, b)

    ker = np.ones(5, np.float32) / 5.0
    sm = np.convolve(mask.astype(np.float32), ker, mode="same")
    mask = sm > 0.5

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0, H, W, H

    y0 = int(idx[0])
    y1 = int(idx[-1]) + 1
    y0 -= y0 % 2
    if (y1 - y0) % 2 == 1:
        y1 += 1
    y0 = max(0, min(y0, H - 2))
    y1 = max(y0 + 2, min(y1, H))
    return y0, y1, W, H


def run_ffmpeg(ff, src, dst, y0, y1, W, H, crf, audio, cb, cancel):
    h = y1 - y0
    h -= h % 2
    y0 -= y0 % 2
    if y0 + h > H:
        h = H - y0
        h -= h % 2
    if h < 2:
        raise ValueError("Gecersiz kirpma alani")

    vf = "crop=%d:%d:0:%d" % (W, h, y0)
    cmd = [ff, "-hide_banner", "-y", "-i", src]
    cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-crf", str(int(crf))]
    cmd += ["-preset", "medium", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += [dst]

    pr = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

    def drain():
        for _ in pr.stderr:
            pass

    threading.Thread(target=drain, daemon=True).start()

    for line in pr.stdout:
        if cancel.is_set():
            pr.terminate()
            break
        line = line.strip()
        if not line.startswith("out_time="):
            continue
        val = line.split("=", 1)[1]
        parts = val.split(":")
        try:
            t = int(parts[0]) * 3600
            t += int(parts[1]) * 60
            t += float(parts[2])
            cb(t)
        except Exception:
            pass

    pr.wait()


class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Video Band Kirpici - ust/alt kirpma")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.files = []
        self.crops = {}
        self.dims = {}
        self.durs = {}
        self.preview = None
        self.photo = None
        self.disp = (1.0, 0, 0)
        self.drag = None
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False

        self.v_mode = tk.StringVar(value="black")
        self.v_sens = tk.IntVar(value=50)
        self.v_top = tk.IntVar(value=0)
        self.v_bot = tk.IntVar(value=0)
        self.v_pos = tk.DoubleVar(value=25.0)
        self.v_crf = tk.IntVar(value=18)
        self.v_sfx = tk.StringVar(value="_kucultulmus")
        self.v_audio = tk.BooleanVar(value=True)

        self.build()
        self.after(150, self.poll)

    # ------------------------------------------------- arayuz
    def build(self):
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=400)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        f1 = ttk.LabelFrame(left, text="1 - Videolar", padding=6)
        f1.pack(fill="x", pady=(0, 6))
        self.lb = tk.Listbox(f1, height=5, selectmode="extended")
        self.lb.pack(fill="x")
        self.lb.bind("<<ListboxSelect>>", lambda e: self.on_select())
        row = ttk.Frame(f1)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Ekle", command=self.add_files).pack(side="left")
        ttk.Button(row, text="Kaldir", command=self.remove_sel).pack(
            side="left", padx=4)
        ttk.Button(row, text="Temizle", command=self.clear_all).pack(side="left")

        f2 = ttk.LabelFrame(left, text="2 - Mod", padding=6)
        f2.pack(fill="x", pady=(0, 6))
        for val, txt in MODES:
            ttk.Radiobutton(
                f2, text=txt, value=val, variable=self.v_mode
            ).pack(anchor="w")
        ttk.Label(f2, text="Hassasiyet (yuksek = daha cok kirpar)").pack(
            anchor="w", pady=(6, 0))
        ttk.Scale(
            f2, from_=0, to=100, variable=self.v_sens, orient="horizontal"
        ).pack(fill="x")
        row = ttk.Frame(f2)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Seciliyi Algila",
                   command=self.detect_sel).pack(side="left")
        ttk.Button(row, text="Tumune Uygula",
                   command=self.detect_all).pack(side="left", padx=4)

        f3 = ttk.LabelFrame(left, text="3 - Sinirlar (surukleyerek sec)",
                            padding=6)
        f3.pack(fill="x", pady=(0, 6))
        row = ttk.Frame(f3)
        row.pack(fill="x")
        ttk.Label(row, text="Ust:").pack(side="left")
        ttk.Spinbox(row, from_=0, to=99999, width=7,
                    textvariable=self.v_top,
                    command=self.on_spin).pack(side="left", padx=4)
        ttk.Label(row, text="Alt:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(row, from_=0, to=99999, width=7,
                    textvariable=self.v_bot,
                    command=self.on_spin).pack(side="left", padx=4)
        self.lbl_cut = ttk.Label(f3, text="-")
        self.lbl_cut.pack(anchor="w", pady=(4, 0))
        self.v_top.trace_add("write", lambda *a: self.on_spin())
        self.v_bot.trace_add("write", lambda *a: self.on_spin())

        f4 = ttk.LabelFrame(left, text="4 - Onizleme karesi", padding=6)
        f4.pack(fill="x", pady=(0, 6))
        ttk.Scale(f4, from_=0, to=100, variable=self.v_pos,
                  orient="horizontal",
                  command=lambda e: self.schedule_preview()).pack(fill="x")

        f5 = ttk.LabelFrame(left, text="5 - Cikti", padding=6)
        f5.pack(fill="x", pady=(0, 6))
        row = ttk.Frame(f5)
        row.pack(fill="x")
        ttk.Label(row, text="Son ek:").pack(side="left")
        ttk.Entry(row, textvariable=self.v_sfx, width=16).pack(
            side="left", padx=4)
        ttk.Label(row, text="CRF:").pack(side="left")
        ttk.Spinbox(row, from_=14, to=30, width=4,
                    textvariable=self.v_crf).pack(side="left", padx=4)
        ttk.Checkbutton(f5, text="Sesi kopyala",
                        variable=self.v_audio).pack(anchor="w", pady=(4, 0))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(6, 4))
        self.btn_run = ttk.Button(row, text="KIRP ve KAYDET",
                                  command=self.start)
        self.btn_run.pack(side="left", fill="x", expand=True)
        self.btn_stop = ttk.Button(row, text="Iptal", state="disabled",
                                   command=self.do_cancel)
        self.btn_stop.pack(side="left", padx=(6, 0))

        self.pbar = ttk.Progressbar(left, maximum=100)
        self.pbar.pack(fill="x")

        self.txt = tk.Text(left, height=7, state="disabled", wrap="word",
                           font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True, pady=(6, 0))

        self.cv = tk.Canvas(right, bg="#15171b", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.bind("<Configure>", lambda e: self.redraw())
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>",
                     lambda e: setattr(self, "drag", None))

        self.lbl_info = ttk.Label(right, text="Bir video secin")
        self.lbl_info.pack(anchor="w", pady=(4, 0))

    def log(self, msg):
        self.txt.config(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.config(state="disabled")

    # ------------------------------------------------- liste
    def add_files(self):
        ps = filedialog.askopenfilenames(
            title="Videolari sec",
            filetypes=[
                ("Videolar", "*.mp4 *.mov *.mkv *.m4v *.avi"),
                ("Tumu", "*.*"),
            ])
        for p in ps:
            if p not in self.files:
                self.files.append(p)
        self.refresh_list()
        if ps:
            i = self.files.index(ps[0])
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(i)
            self.on_select()

    def remove_sel(self):
        for i in reversed(self.lb.curselection()):
            p = self.files.pop(i)
            self.crops.pop(p, None)
            self.dims.pop(p, None)
            self.durs.pop(p, None)
        self.refresh_list()

    def clear_all(self):
        self.files = []
        self.crops = {}
        self.dims = {}
        self.durs = {}
        self.preview = None
        self.refresh_list()
        self.redraw()

    def refresh_list(self):
        sel = list(self.lb.curselection())
        self.lb.delete(0, "end")
        for p in self.files:
            name = os.path.basename(p)
            if p in self.crops and p in self.dims:
                y0, y1 = self.crops[p]
                H = self.dims[p][1]
                name += "   [ust %d / alt %d]" % (y0, H - y1)
            self.lb.insert("end", name)
        for i in sel:
            if i < self.lb.size():
                self.lb.selection_set(i)

    def cur_file(self):
        sel = self.lb.curselection()
        if sel:
            return self.files[sel[0]]
        if self.files:
            return self.files[0]
        return None

    def load_info(self, p):
        cap, tmp = open_cap(p)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        rm(tmp)
        self.dims[p] = (W, H)
        if fc > 0 and fps > 0:
            self.durs[p] = fc / fps
        else:
            self.durs[p] = 0.0

    def on_select(self):
        p = self.cur_file()
        if not p:
            return
        if p not in self.dims:
            try:
                self.load_info(p)
            except Exception as e:
                self.log("HATA: %s" % e)
                return
        W, H = self.dims[p]
        if p not in self.crops:
            self.crops[p] = (0, H)
        y0, y1 = self.crops[p]
        self.v_top.set(y0)
        self.v_bot.set(H - y1)
        self.show_preview()

    def on_spin(self):
        p = self.cur_file()
        if (not p) or (p not in self.dims):
            return
        W, H = self.dims[p]
        try:
            t = int(self.v_top.get())
            b = int(self.v_bot.get())
        except Exception:
            return
        t = max(0, min(t, H - 2))
        b = max(0, min(b, H - 2))
        y1 = H - b
        if y1 - t < 2:
            return
        self.crops[p] = (t, y1)
        self.lbl_cut.config(
            text="Ust %d px / Alt %d px / Sonuc %dx%d" % (t, b, W, y1 - t))
        self.redraw()

    # ------------------------------------------------- onizleme
    def schedule_preview(self):
        if hasattr(self, "job"):
            self.after_cancel(self.job)
        self.job = self.after(250, self.show_preview)

    def show_preview(self):
        p = self.cur_file()
        if not p:
            return
        try:
            cap, tmp = open_cap(p)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = int(self.v_pos.get() / 100.0 * max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = cap.read()
            cap.release()
            rm(tmp)
        except Exception as e:
            self.log("Onizleme hatasi: %s" % e)
            return
        if ok:
            self.preview = fr
            self.dims[p] = (fr.shape[1], fr.shape[0])
            if p not in self.crops:
                self.crops[p] = (0, fr.shape[0])
            self.redraw()

    def redraw(self):
        self.cv.delete("all")
        p = self.cur_file()
        if (self.preview is None) or (not p):
            return
        H, W = self.preview.shape[:2]
        cw = max(self.cv.winfo_width() - 20, 100)
        ch = max(self.cv.winfo_height() - 20, 100)
        sc = min(float(cw) / W, float(ch) / H)
        dw = max(int(W * sc), 1)
        dh = max(int(H * sc), 1)
        ox = (self.cv.winfo_width() - dw) // 2
        oy = (self.cv.winfo_height() - dh) // 2
        self.disp = (sc, ox, oy)

        small = cv2.resize(self.preview, (dw, dh),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ok, png = cv2.imencode(".png", rgb)
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.cv.create_image(ox, oy, anchor="nw", image=self.photo)

        y0, y1 = self.crops.get(p, (0, H))
        ty = oy + y0 * sc
        by = oy + y1 * sc

        if y0 > 0:
            self.cv.create_rectangle(ox, oy, ox + dw, ty,
                                     fill="#e0342c", stipple="gray25",
                                     outline="")
        if y1 < H:
            self.cv.create_rectangle(ox, by, ox + dw, oy + dh,
                                     fill="#e0342c", stipple="gray25",
                                     outline="")

        self.cv.create_line(ox, ty, ox + dw, ty, fill="#38d39f", width=2)
        self.cv.create_line(ox, by, ox + dw, by, fill="#38d39f", width=2)
        self.cv.create_text(ox + 8, max(ty - 12, oy + 4), anchor="w",
                            fill="#38d39f", text="UST y=%d" % y0,
                            font=("TkDefaultFont", 9, "bold"))
        self.cv.create_text(ox + 8, min(by + 14, oy + dh - 4), anchor="w",
                            fill="#38d39f", text="ALT y=%d" % y1,
                            font=("TkDefaultFont", 9, "bold"))
        self.cv.create_rectangle(ox, oy, ox + dw, oy + dh, outline="#555a63")

        self.lbl_info.config(
            text="%s   %dx%d   sonuc %dx%d   (yesil cizgileri surukleyin)"
            % (os.path.basename(p), W, H, W, max(y1 - y0, 0)))

    # ------------------------------------------------- fare
    def on_press(self, e):
        p = self.cur_file()
        if (not p) or (self.preview is None):
            return
        sc, ox, oy = self.disp
        H, W = self.preview.shape[:2]
        if not (ox <= e.x <= ox + W * sc):
            return
        if not (oy <= e.y <= oy + H * sc):
            return
        yi = (e.y - oy) / sc
        y0, y1 = self.crops.get(p, (0, H))
        if abs(yi - y0) <= abs(yi - y1):
            self.drag = "top"
        else:
            self.drag = "bot"
        self.on_drag(e)

    def on_drag(self, e):
        p = self.cur_file()
        if (not p) or (self.drag is None) or (self.preview is None):
            return
        sc, ox, oy = self.disp
        H, W = self.preview.shape[:2]
        y = int(round((e.y - oy) / sc))
        y = max(0, min(y, H))
        y0, y1 = self.crops.get(p, (0, H))
        if self.drag == "top":
            y0 = min(max(0, y), y1 - 2)
        else:
            y1 = max(min(H, y), y0 + 2)
        self.crops[p] = (y0, y1)
        self.v_top.set(y0)
        self.v_bot.set(H - y1)
        self.redraw()

    # ------------------------------------------------- algilama
    def detect_sel(self):
        p = self.cur_file()
        if p:
            self.detect([p])

    def detect_all(self):
        if self.files:
            self.detect(list(self.files))

    def detect(self, paths):
        if self.busy:
            return
        mode = self.v_mode.get()
        if mode == "manual":
            self.log("Elle modda algilama yok, cizgileri surukleyin.")
            return
        cfg = {"mode": mode, "sens": int(self.v_sens.get())}
        self.busy = True
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.cancel.clear()
        threading.Thread(target=self.detect_work,
                         args=(paths, cfg),
                         daemon=True).start()

    def detect_work(self, paths, cfg):
        try:
            for p in paths:
                if self.cancel.is_set():
                    break
                self.q.put(("log", "Analiz: " + os.path.basename(p)))
                try:
                    y0, y1, W, H = analyze(p, cfg["mode"], cfg["sens"])
                    self.q.put(("dim", (p, W, H)))
                    self.q.put(("crop", (p, y0, y1)))
                    self.q.put((
                        "log",
                        "  ust %d / alt %d / sonuc %dx%d" % (
                            y0, H - y1, W, y1 - y0),
                    ))
                except Exception as e:
                    self.q.put(("log", "  HATA: %s" % e))
        finally:
            self.q.put(("done", None))

    # ------------------------------------------------- isleme
    def start(self):
        if self.busy or (not self.files):
            return
        mode = self.v_mode.get()
        if (mode == "manual") and (len(self.files) > 1):
            ok = messagebox.askyesno(
                "Toplu kirpma",
                "Elle degerler TUM videolara ayni sekilde uygulanacak.\n"
                "Devam edilsin mi?")
            if not ok:
                return
        cfg = {
            "mode": mode,
            "sens": int(self.v_sens.get()),
            "top": int(self.v_top.get()),
            "bot": int(self.v_bot.get()),
            "crf": int(self.v_crf.get()),
            "audio": bool(self.v_audio.get()),
            "sfx": self.v_sfx.get().strip() or "_kucultulmus",
        }
        self.busy = True
        self.cancel.clear()
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.pbar["value"] = 0
        threading.Thread(target=self.run_work,
                         args=(cfg,),
                         daemon=True).start()

    def do_cancel(self):
        self.cancel.set()

    def out_path(self, src, sfx):
        base, _ = os.path.splitext(src)
        dst = base + sfx + ".mp4"
        k = 2
        while os.path.exists(dst):
            dst = "%s%s_%d.mp4" % (base, sfx, k)
            k += 1
        return dst

    def run_work(self, cfg):
        ff = find_ffmpeg()
        if not ff:
            self.q.put(("log", "HATA: ffmpeg bulunamadi."))
            self.q.put(("log", "Cozum: pip install imageio-ffmpeg"))
            self.q.put(("done", None))
            return

        n = len(self.files)
        done = 0

        for i, src in enumerate(self.files):
            if self.cancel.is_set():
                self.q.put(("log", "Iptal edildi."))
                break

            self.q.put((
                "log",
                "[%d/%d] %s" % (i + 1, n, os.path.basename(src)),
            ))

            try:
                self.load_info(src)
                W, H = self.dims[src]

                if cfg["mode"] == "manual":
                    y0 = cfg["top"]
                    y1 = H - cfg["bot"]
                    y0 = max(0, min(y0, H - 2))
                    y1 = max(y0 + 2, min(y1, H))
                else:
                    y0, y1, W, H = analyze(src, cfg["mode"], cfg["sens"])
                    self.q.put(("dim", (src, W, H)))
                    self.q.put(("crop", (src, y0, y1)))

                self.q.put((
                    "log",
                    "  kirp: ust %d px, alt %d px" % (y0, H - y1),
                ))

                dst = self.out_path(src, cfg["sfx"])
                dur = self.durs.get(src, 0.0)

                def cb(t, i=i, dur=dur):
                    if dur > 0:
                        frac = min(t / dur, 1.0)
                        self.q.put(("prog", 100.0 * (i + frac) / n))

                run_ffmpeg(ff, src, dst, y0, y1, W, H,
                           cfg["crf"], cfg["audio"], cb, self.cancel)

                if self.cancel.is_set():
                    rm(dst)
                    self.q.put(("log", "  iptal edildi."))
                    break

                self.q.put(("prog", 100.0 * (i + 1) / n))
                self.q.put(("log", "  Kaydedildi: " + os.path.basename(dst)))
                done += 1

            except Exception as e:
                self.q.put(("log", "  HATA: %s" % e))

        self.q.put(("log", "Bitti: %d/%d dosya islendi." % (done, n)))
        self.q.put(("done", None))

    # ------------------------------------------------- kuyruk
    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log(val)
                elif kind == "prog":
                    self.pbar["value"] = val
                elif kind == "dim":
                    p, W, H = val
                    self.dims[p] = (W, H)
                elif kind == "crop":
                    p, y0, y1 = val
                    self.crops[p] = (y0, y1)
                    if p == self.cur_file():
                        H = self.dims.get(p, (0, 0))[1]
                        self.v_top.set(y0)
                        self.v_bot.set(H - y1)
                        self.redraw()
                elif kind == "done":
                    self.busy = False
                    self.btn_run.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.refresh_list()
        except queue.Empty:
            pass
        self.after(100, self.poll)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()