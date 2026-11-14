"""Prepare ref_audio.wav (24 kHz mono) from the local human Thai voice seed (or edge-tts fallback).

Run once: python generate_ref.py
Or replace ref_audio.wav with any 3-8 s clean 24 kHz mono recording and update REF_TEXT in config.py.
"""
import asyncio
import os
import tempfile

import edge_tts
import librosa
import numpy as np
import soundfile as sf

import config as C


async def _load_source_audio():
    if os.path.exists(C.REF_SEED_PATH):
        wav, _ = librosa.load(C.REF_SEED_PATH, sr=C.TTS_SAMPLE_RATE, mono=True)
        return wav, C.REF_SEED_PATH
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        await edge_tts.Communicate(C.REF_TEXT, C.REF_VOICE).save(tmp)
        wav, _ = librosa.load(tmp, sr=C.TTS_SAMPLE_RATE, mono=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return wav, f"edge-tts:{C.REF_VOICE}"


async def main():
    wav, src = await _load_source_audio()
    wav, _ = librosa.effects.trim(wav, top_db=35)
    wav = np.concatenate([wav, np.zeros(int(0.10 * C.TTS_SAMPLE_RATE), dtype=np.float32)])
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav * (0.85 / peak)
    sf.write(C.REF_AUDIO_PATH, wav.astype(np.float32), C.TTS_SAMPLE_RATE, subtype="PCM_16")
    info = sf.info(C.REF_AUDIO_PATH)
    print(f"Wrote {C.REF_AUDIO_PATH} from {src} ({info.duration:.2f} s, {info.samplerate} Hz, {info.channels} ch)")


if __name__ == "__main__":
    asyncio.run(main())
