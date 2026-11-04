"""Create ref_audio.wav (24 kHz mono) for voice cloning using edge-tts.

Run once: python generate_ref.py
To use your own voice instead, just overwrite ref_audio.wav and edit REF_TEXT in config.py.
"""
import asyncio
import os
import tempfile

import edge_tts
import librosa
import numpy as np
import soundfile as sf

from config import REF_AUDIO_PATH, REF_TEXT, REF_VOICE, TTS_SAMPLE_RATE


async def main():
    mp3 = os.path.join(tempfile.gettempdir(), "ref_edge_tts.mp3")
    # rate -10% makes this sentence ~4 s, a good length for reference audio
    await edge_tts.Communicate(REF_TEXT, REF_VOICE, rate="-10%").save(mp3)

    wav, _ = librosa.load(mp3, sr=TTS_SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=35)
    # a short silent tail after the reference speech, as cloning models expect
    wav = np.concatenate([wav, np.zeros(int(0.3 * TTS_SAMPLE_RATE), dtype=wav.dtype)])
    wav = 0.9 * wav / (np.abs(wav).max() + 1e-9)

    sf.write(REF_AUDIO_PATH, wav, TTS_SAMPLE_RATE, subtype="PCM_16")
    os.remove(mp3)
    print(f"saved {REF_AUDIO_PATH} ({len(wav) / TTS_SAMPLE_RATE:.2f} s, {TTS_SAMPLE_RATE} Hz mono)")


if __name__ == "__main__":
    asyncio.run(main())
