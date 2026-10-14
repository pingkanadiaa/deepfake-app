import os
import time
import tempfile

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio
import timm
import streamlit as st
from PIL import Image

# ============================================================
# Konfigurasi (HARUS sama dengan saat training)
# ============================================================
IMG_SIZE  = 224
N_FRAMES  = 8
SR        = 16_000
AUDIO_LEN = SR * 4            # 4 detik = 64.000 sampel
VIT_DIM, AUD_DIM, INV_DIM = 192, 256, 64
ATT_HEADS, MLP_DIM, DROPOUT, NUM_CLS = 4, 128, 0.1, 4

CLASSES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
           "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

MODEL_FILES = {
    "MIS-AVoiDD (Model Utama)":        "models/best_model_misavoidd.pth",
    "Direct Concatenation (Baseline)": "models/best_model_baseline.pth",
}
SAMPLES_DIR = "samples"       # samples/<nama_kelas>/*.mp4

# Bobot model diambil dari Hugging Face jika file lokal tidak ada.
# GANTI dengan repo kamu, format "username/nama-repo".
HF_REPO = "pingkannadia/deepfake-model-weights"
# Dataset contoh uji di Hugging Face (repo tipe Dataset, boleh Private).
# Struktur: <nama_kelas>/<video>.mp4. Kosongkan ("") jika hanya pakai folder lokal 'samples/'.
HF_SAMPLES_REPO = "pingkannadia/deepfake-samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Arsitektur model (sama dengan notebook training)
# ============================================================
class VisualBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)

    def forward(self, x):
        return self.vit(x)


class AudioBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256, AUD_DIM), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.head(self.net(x))


class MISAVoiDD(nn.Module):
    def __init__(self):
        super().__init__()
        D = INV_DIM
        self.v_proj = nn.Linear(VIT_DIM, D)
        self.a_proj = nn.Linear(AUD_DIM, D)
        self.Tv = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Dropout(DROPOUT))
        self.Ta = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Dropout(DROPOUT))
        self.Ec = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Dropout(DROPOUT))
        # decoder hanya dipakai saat training, tetap didefinisikan agar bobot bisa dimuat
        self.dec_v = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.dec_a = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.mha = nn.MultiheadAttention(embed_dim=D, num_heads=ATT_HEADS,
                                         dropout=DROPOUT, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(D * 4, MLP_DIM), nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT), nn.Linear(MLP_DIM, NUM_CLS),
        )

    def forward(self, x_v, x_a):
        xv, xa = self.v_proj(x_v), self.a_proj(x_a)
        h_v, h_a = self.Tv(xv), self.Ta(xa)
        g_v, g_a = self.Ec(xv), self.Ec(xa)
        M = torch.stack([h_a, h_v, g_a, g_v], dim=1)
        M_bar, _ = self.mha(M, M, M)
        return self.classifier(M_bar.flatten(1))


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_backbone = VisualBackbone()
        self.a_backbone = AudioBackbone()
        self.mis = MISAVoiDD()

    def forward(self, frames, waveform):
        B, T_, C, H, W = frames.shape
        fv = self.v_backbone(frames.view(B * T_, C, H, W))
        x_v = fv.view(B, T_, -1).mean(dim=1)
        return self.mis(x_v, self.a_backbone(waveform))


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_backbone = VisualBackbone()
        self.a_backbone = AudioBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(VIT_DIM + AUD_DIM, MLP_DIM), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(MLP_DIM, MLP_DIM), nn.ReLU(inplace=True), nn.Dropout(DROPOUT),
            nn.Linear(MLP_DIM, NUM_CLS),
        )

    def forward(self, frames, waveform):
        B, T_, C, H, W = frames.shape
        fv = self.v_backbone(frames.view(B * T_, C, H, W))
        x_v = fv.view(B, T_, -1).mean(dim=1)
        return self.classifier(torch.cat([x_v, self.a_backbone(waveform)], dim=-1))


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_name):
    model = FullModel() if model_name.startswith("MIS") else BaselineModel()
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        try:
            token = st.secrets["HF_TOKEN"]
        except Exception:
            token = None          # repo publik: tidak perlu token
        path = hf_hub_download(HF_REPO, os.path.basename(path), token=token)
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        size_mb = os.path.getsize(path) / 1e6
        raise RuntimeError(f"Gagal membaca bobot '{os.path.basename(path)}' ({size_mb:.2f} MB): {e}")
    if isinstance(state, dict) and "model_state" in state:   # jika memakai file checkpoint_epXX.pth
        state = state["model_state"]
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


# ============================================================
# Praproses (sama dengan training, tanpa augmentasi)
# ============================================================
_norm = T.Compose([T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def extract_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(0, total - 1, N_FRAMES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                f = cv2.resize(f, (IMG_SIZE, IMG_SIZE))
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)          # grayscale
                frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))  # direplikasi 3 kanal
    cap.release()
    if not frames:
        return None
    while len(frames) < N_FRAMES:
        frames.append(frames[-1].copy())
    tensors = [_norm(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames[:N_FRAMES]]
    return torch.stack(tensors).unsqueeze(0)                     # (1, T, 3, H, W)


def _load_audio_ffmpeg(path):
    """Ekstrak audio mono 16 kHz (maks 4 dtk) lewat ffmpeg. Paling andal di Streamlit Cloud."""
    import subprocess, shutil
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-t", "4", "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def extract_waveform(path):
    y = None
    for loader in ("ffmpeg", "torchaudio", "librosa"):
        try:
            if loader == "ffmpeg":
                y = _load_audio_ffmpeg(path)
            elif loader == "torchaudio":
                wav, sr = torchaudio.load(path, format="mp4")
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != SR:
                    wav = torchaudio.transforms.Resample(sr, SR)(wav)
                y = wav.squeeze(0).numpy().astype(np.float32)
            else:
                import librosa
                y, _ = librosa.load(path, sr=SR, mono=True)
                y = y.astype(np.float32)
            if y is not None and len(y) > 0:
                break
        except Exception:
            y = None
    if y is None:
        return None
    if len(y) == 0:
        return None
    y = np.pad(y, (0, AUDIO_LEN - len(y))) if len(y) < AUDIO_LEN else y[:AUDIO_LEN]
    return torch.from_numpy(y).view(1, 1, -1)                    # (1, 1, 64000)


def run_inference(model, frames, wave):
    frames, wave = frames.to(DEVICE), wave.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = F.softmax(model(frames, wave).float(), dim=-1)[0].cpu().numpy()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return probs, time.perf_counter() - t0


# ============================================================
# Antarmuka
# ============================================================
st.set_page_config(page_title="Klasifikasi Deepfake Audio-Visual", layout="wide")

with st.sidebar:
    st.header("Panel Kontrol")
    mode = st.radio("Mode Aktif", ["Klasifikasi", "Evaluasi"])
    model_name = st.radio("Model Aktif", list(MODEL_FILES))
    st.markdown("**Legenda Kelas**")
    for c in CLASSES:
        st.markdown(f"- {c}")
    st.caption(f"Perangkat: {DEVICE.upper()}")

st.title("Klasifikasi Deepfake Audio-Visual")

sumber = st.radio("Sumber video", ["Contoh data uji", "Unggah video"], horizontal=True)
video_path, actual = None, None

def _hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def list_samples():
    """Return {kelas: [nama_file,...]} dari folder lokal atau dataset HF."""
    out = {}
    if os.path.isdir(SAMPLES_DIR):
        for c in CLASSES:
            d = os.path.join(SAMPLES_DIR, c)
            if os.path.isdir(d):
                out[c] = sorted(f for f in os.listdir(d) if f.lower().endswith(".mp4"))
    if not out and HF_SAMPLES_REPO and not HF_SAMPLES_REPO.startswith("USERNAME"):
        from huggingface_hub import list_repo_files
        for p in list_repo_files(HF_SAMPLES_REPO, repo_type="dataset", token=_hf_token()):
            parts = p.split("/")
            if len(parts) == 2 and parts[0] in CLASSES and p.lower().endswith(".mp4"):
                out.setdefault(parts[0], []).append(parts[1])
        out = {k: sorted(v) for k, v in out.items()}
    return out

def get_sample_path(cls, fname):
    local = os.path.join(SAMPLES_DIR, cls, fname)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_SAMPLES_REPO, f"{cls}/{fname}", repo_type="dataset", token=_hf_token())

if sumber == "Contoh data uji":
    samples = list_samples()
    if not samples:
        st.info("Contoh data uji belum tersedia. Gunakan 'Unggah video'.")
    else:
        col1, col2 = st.columns(2)
        cls = col1.selectbox("Kelas data uji", list(samples))
        fname = col2.selectbox("Video", samples[cls])
        video_path = get_sample_path(cls, fname)
        actual = cls
else:
    up = st.file_uploader("Unggah file video (.mp4)", type=["mp4"])
    if up is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(up.read())
        tmp.close()
        video_path = tmp.name
    if mode == "Evaluasi":
        actual = st.selectbox("Label aktual video", CLASSES)

if video_path:
    st.video(video_path)
    if st.button("Klasifikasi", type="primary"):
        with st.spinner("Memproses video..."):
            t0 = time.perf_counter()
            frames = extract_frames(video_path)
            wave = extract_waveform(video_path)
            t_pre = time.perf_counter() - t0

            if frames is None:
                st.error("Frame video tidak dapat dibaca. Pastikan file video valid.")
                st.stop()
            if wave is None:
                st.warning("Audio tidak tersedia pada video ini, sehingga audio diganti sinyal kosong "
                           "(hasil prediksi mungkin kurang akurat).")
                wave = torch.zeros(1, 1, AUDIO_LEN)

            probs, t_model = run_inference(load_model(model_name), frames, wave)

        pred = CLASSES[int(np.argmax(probs))]
        st.subheader("Hasil Klasifikasi")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kelas Prediksi", pred)
        m2.metric("Probabilitas", f"{probs.max() * 100:.2f}%")
        m3.metric("Waktu Inferensi Model", f"{t_model * 1000:.1f} ms")
        m4.metric("Waktu Praproses", f"{t_pre:.2f} detik")

        st.bar_chart(pd.DataFrame({"Probabilitas": probs}, index=CLASSES))

        if mode == "Evaluasi" and actual is not None:
            st.subheader("Evaluasi")
            st.write(f"Label aktual: **{actual}**")
            if pred == actual:
                st.success("Prediksi SESUAI dengan label aktual.")
            else:
                st.error("Prediksi TIDAK SESUAI dengan label aktual.")
else:
    st.caption("Pilih atau unggah video terlebih dahulu.")


# ============================================================
# Evaluasi batch: seluruh contoh data uji dari keempat kelas
# ============================================================
@st.cache_data(show_spinner=False)
def evaluate_batch(model_name, samples_json):
    import json
    samples_ = json.loads(samples_json)
    items = [(c, f) for c, fs in samples_.items() for f in fs]
    model = load_model(model_name)
    rows = []
    bar = st.progress(0.0, text="Memproses video...")
    for i, (c, f) in enumerate(items):
        path = get_sample_path(c, f)
        frames = extract_frames(path)
        wave = extract_waveform(path)
        if frames is None:
            rows.append({"Video": f, "Label Aktual": c, "Prediksi": "(gagal dibaca)",
                         "Probabilitas (%)": 0.0, "Hasil": "Gagal", "Waktu Model (ms)": 0.0})
        else:
            if wave is None:
                wave = torch.zeros(1, 1, AUDIO_LEN)
            probs, tm = run_inference(model, frames, wave)
            pred = CLASSES[int(np.argmax(probs))]
            rows.append({"Video": f, "Label Aktual": c, "Prediksi": pred,
                         "Probabilitas (%)": round(float(probs.max()) * 100, 2),
                         "Hasil": "Benar" if pred == c else "Salah",
                         "Waktu Model (ms)": round(tm * 1000, 1)})
        bar.progress((i + 1) / len(items), text=f"Memproses video {i + 1}/{len(items)}")
    bar.empty()
    return pd.DataFrame(rows)


def show_batch_result(df):
    valid = df[df["Hasil"] != "Gagal"]
    acc = (valid["Hasil"] == "Benar").mean() * 100 if len(valid) else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi", f"{acc:.2f}%")
    c2.metric("Jumlah Video", len(df))
    c3.metric("Rata-rata Waktu Model", f"{valid['Waktu Model (ms)'].mean():.1f} ms" if len(valid) else "-")

    st.markdown("**Ringkasan per kelas**")
    per = valid.groupby("Label Aktual").agg(
        Total=("Hasil", "size"), Benar=("Hasil", lambda x: int((x == "Benar").sum())))
    per["Akurasi (%)"] = (per["Benar"] / per["Total"] * 100).round(2)
    st.dataframe(per.reindex([c for c in CLASSES if c in per.index]), use_container_width=True)

    st.markdown("**Confusion matrix** (baris = aktual, kolom = prediksi)")
    cm = pd.crosstab(valid["Label Aktual"], valid["Prediksi"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    st.dataframe(cm, use_container_width=True)

    st.markdown("**Detail per video**")
    st.dataframe(df, use_container_width=True, hide_index=True)


if mode == "Evaluasi":
    st.divider()
    st.header("Evaluasi Seluruh Contoh Data Uji")
    all_samples = list_samples()
    if not all_samples:
        st.info("Contoh data uji belum tersedia.")
    else:
        import json
        n_vid = sum(len(v) for v in all_samples.values())
        st.caption(f"{n_vid} video dari {len(all_samples)} kelas.")
        bandingkan = st.checkbox("Bandingkan kedua model")
        if st.button("Evaluasi Semua Kelas", type="primary"):
            key = json.dumps(all_samples, sort_keys=True)
            if bandingkan:
                cols = st.columns(2)
                for col, name in zip(cols, MODEL_FILES):
                    with col:
                        st.subheader(name)
                        show_batch_result(evaluate_batch(name, key))
            else:
                st.subheader(model_name)
                show_batch_result(evaluate_batch(model_name, key))
