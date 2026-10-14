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
        token = st.secrets.get("HF_TOKEN", None) if hasattr(st, "secrets") else None
        path = hf_hub_download(HF_REPO, os.path.basename(path), token=token)
    state = torch.load(path, map_location=DEVICE)
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


def extract_waveform(path):
    try:
        wav, sr = torchaudio.load(path, format="mp4")
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.transforms.Resample(sr, SR)(wav)
        y = wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        try:
            import librosa
            y, _ = librosa.load(path, sr=SR, mono=True)
            y = y.astype(np.float32)
        except Exception:
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

if sumber == "Contoh data uji":
    avail = [c for c in CLASSES if os.path.isdir(os.path.join(SAMPLES_DIR, c))]
    if not avail:
        st.info(f"Folder '{SAMPLES_DIR}/<nama_kelas>/' belum berisi video contoh.")
    else:
        col1, col2 = st.columns(2)
        cls = col1.selectbox("Kelas data uji", avail)
        files = sorted(f for f in os.listdir(os.path.join(SAMPLES_DIR, cls)) if f.lower().endswith(".mp4"))
        fname = col2.selectbox("Video", files)
        video_path = os.path.join(SAMPLES_DIR, cls, fname)
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
