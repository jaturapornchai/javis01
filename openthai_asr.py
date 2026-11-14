"""Streaming Thai + mixed English ASR using iApp's OpenThai 2.0 ASR Streaming (ONNX)."""
from __future__ import annotations

import os
import threading
import warnings
from typing import Optional

import librosa
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LEN = 400
HOP = 160
HALF_WIN = N_FFT // 2
BLANK_ID = 14988
MAX_SYMS_PER_STEP = 2
MAX_SEGMENT_S = 12.0  # cut continuous streams at a quiet dip to keep latency < 250 ms
PROMPTS = {"th-TH": 32, "th-mix": 63}


def display_text(text: str) -> str:
    """Recompose decomposed SARA AM (U+0E4D U+0E32 -> U+0E33) and tidy spacing."""
    return " ".join(text.replace("\u0e4d\u0e32", "\u0e33").split())


def _load_tokens(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\r\n") for ln in f]


def _hann_512() -> np.ndarray:
    """Symmetric 400-sample Hann window centered in N_FFT=512 (matches NeMo / torch.stft)."""
    w = np.zeros(N_FFT, dtype=np.float32)
    off = (N_FFT - WIN_LEN) // 2
    w[off:off + WIN_LEN] = np.hanning(WIN_LEN).astype(np.float32)
    return w


class OpenThaiASR:
    """Thread-safe CPU ONNX runner for OpenThai2.0-ASR-Streaming-ONNX."""

    def __init__(self, model_dir: str, lang: str = "th-TH", intra_threads: int = 0) -> None:
        if lang not in PROMPTS:
            raise ValueError(f"Unsupported lang {lang!r}; expected one of {tuple(PROMPTS)}")
        self.model_dir = model_dir
        self.lang = lang
        self.prompt_id = np.array([PROMPTS[lang]], dtype=np.int64)
        self.tokens = _load_tokens(os.path.join(model_dir, "tokens.txt"))
        self._win = _hann_512()
        self._mel_fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney", htk=False
        ).astype(np.float32)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads or min(8, os.cpu_count() or 4)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._enc = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self._dec = ort.InferenceSession(os.path.join(model_dir, "decoder_joint.onnx"), opts, providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    def log_mel(self, audio_16k: np.ndarray) -> np.ndarray:
        """Compute [1, 128, T] unnormalized NeMo-compatible log-mel spectrogram."""
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros((1, 128, 0), dtype=np.float32)
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - 0.97 * x[:-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_fft=.*is too large")
            stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=self._win, center=True, pad_mode="constant")
        power = (stft.real ** 2 + stft.imag ** 2).astype(np.float32)
        return np.log(self._mel_fb @ power + (2.0 ** -24)).astype(np.float32)[None, :, :]

    def _encode(self, mel: np.ndarray) -> np.ndarray:
        n_frames = mel.shape[2]
        if n_frames == 0:
            return np.zeros((1, 0, 1024), dtype=np.float32)
        length = np.array([n_frames], dtype=np.int64)
        with self._lock:
            encoded, enc_len = self._enc.run(
                None, {"audio_signal": mel, "length": length, "prompt_index": self.prompt_id}
            )
        return encoded[:, :int(enc_len[0]), :]

    def _decode_encoded(self, encoded: np.ndarray, delta: float) -> str:
        t_enc = encoded.shape[1]
        if t_enc == 0:
            return ""
        s1 = np.zeros((2, 1, 640), dtype=np.float32)
        s2 = np.zeros((2, 1, 640), dtype=np.float32)
        tgt = np.array([[BLANK_ID]], dtype=np.int32)
        tlen = np.array([1], dtype=np.int32)
        ids: list[int] = []
        with self._lock:
            for t in range(t_enc):
                frame = encoded[:, t:t + 1, :].transpose(0, 2, 1)
                for _ in range(MAX_SYMS_PER_STEP):
                    logits, _, ns1, ns2 = self._dec.run(
                        None,
                        {"encoder_outputs": frame, "targets": tgt, "target_length": tlen,
                         "input_states_1": s1, "input_states_2": s2},
                    )
                    scores = logits.reshape(-1).copy()
                    scores[BLANK_ID] += delta
                    k = int(np.argmax(scores))
                    if k == BLANK_ID:
                        break
                    ids.append(k)
                    tgt = np.array([[k]], dtype=np.int32)
                    s1, s2 = ns1, ns2
        pieces = [
            self.tokens[i] for i in ids
            if 0 <= i < len(self.tokens) and not (self.tokens[i].startswith("<") and self.tokens[i].endswith(">"))
        ]
        return display_text("".join(pieces).replace("▁", " "))

    def decode_mel(self, mel: np.ndarray, audio_s: float) -> str:
        """Decode precomputed log-mel features with iApp's shortclip blank-bias policy."""
        if mel.shape[2] == 0:
            return ""
        encoded = self._encode(mel)
        delta = -2.5 if audio_s < 2.0 else -1.0
        text = self._decode_encoded(encoded, delta=delta)
        if not text and audio_s >= 0.25:
            text = self._decode_encoded(encoded, delta=-4.0)
        return text

    def transcribe(self, audio_16k: np.ndarray) -> str:
        """Transcribe a 16 kHz mono float32 utterance cleanly."""
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if x.size < 800 or not np.any(x):
            return ""
        return self.decode_mel(self.log_mel(x), x.size / SAMPLE_RATE)

    def stream(self) -> "OpenThaiStream":
        """Create an incremental streaming session for one live utterance."""
        return OpenThaiStream(self)


class OpenThaiStream:
    """Incremental single-utterance streaming decoder with feature-frame caching."""

    def __init__(self, asr: OpenThaiASR) -> None:
        self.asr = asr
        self._audio = np.zeros(0, dtype=np.float32)
        self._feats: Optional[np.ndarray] = None
        self._committed = 0
        self._prefix = ""
        self._partial = ""
        self._last_dec_frames = 0
        self._done = False
        self._lock = threading.Lock()

    def _commit(self, final: bool) -> int:
        n = self._audio.size
        if final:
            upto = n // HOP + 1 if n > 0 else 0
        else:
            upto = (n - HALF_WIN) // HOP + 1 if n >= HALF_WIN else 0
        if upto <= self._committed:
            return self._committed
        f0 = self._committed
        s = 0 if f0 < 3 else (f0 - 2) * HOP
        seg_mel = self.asr.log_mel(self._audio[s:n])
        j0, j1 = f0 - s // HOP, upto - s // HOP
        new_frames = seg_mel[:, :, j0:j1]
        self._feats = new_frames if self._feats is None else np.concatenate((self._feats, new_frames), axis=2)
        self._committed = upto
        return upto

    def _roll_if_long(self) -> None:
        """Cut very long continuous speech at a local energy dip to bound encoder latency."""
        max_samples = int(MAX_SEGMENT_S * SAMPLE_RATE)
        if self._audio.size <= max_samples:
            return
        lo, hi = int(7.5 * SAMPLE_RATE), int(11.0 * SAMPLE_RATE)
        win = 640  # 40 ms RMS window
        energy = [float(np.mean(self._audio[p:p + win] ** 2)) for p in range(lo, hi, HOP)]
        cut_sample = lo + int(np.argmin(energy)) * HOP
        cut_frame = cut_sample // HOP
        seg_text = self.asr.decode_mel(self._feats[:, :, :cut_frame], cut_sample / SAMPLE_RATE)
        if seg_text:
            self._prefix = f"{self._prefix} {seg_text}".strip()
        tail = self._audio[cut_sample:].copy()
        self._audio = tail
        self._feats = None
        self._committed = 0
        self._last_dec_frames = 0
        self._commit(final=False)

    def _combined(self, text: str) -> str:
        return " ".join(p for p in (self._prefix, text) if p)

    def accept(self, pcm_16k: np.ndarray) -> str:
        """Append 16 kHz mono float32 samples and return updated live transcript."""
        x = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._done:
                raise RuntimeError("stream already finished")
            if x.size:
                self._audio = np.concatenate((self._audio, x)) if self._audio.size else x.copy()
                self._commit(final=False)
                self._roll_if_long()
            if self._feats is not None and self._committed >= 25 and self._committed > self._last_dec_frames:
                self._last_dec_frames = self._committed
                tail_text = self.asr.decode_mel(self._feats, self._audio.size / SAMPLE_RATE)
                self._partial = self._combined(tail_text)
            return self._partial

    def finish(self) -> str:
        """Flush final edge frames and return the complete utterance transcript."""
        with self._lock:
            if self._done:
                return self._partial
            self._done = True
            if self._audio.size < 800 or not np.any(self._audio):
                return self._prefix
            self._commit(final=True)
            tail_text = self.asr.decode_mel(self._feats, self._audio.size / SAMPLE_RATE) if self._feats is not None else ""
            self._partial = self._combined(tail_text)
            return self._partial
