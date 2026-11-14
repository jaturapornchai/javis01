"""Local real-time Thai voice assistant (phone-call style secretary).

Browser (static/index.html: mic over WebRTC, SVG status ring)
    -> FastRTC (Silero VAD pause detection, barge-in) -> OpenThai 2.0 ASR Streaming (CPU ONNX STT)
    -> DeepSeek streaming + tools (web, docs, bcbrain MCP)
    -> spoken summary -> F5-TTS-THAI (local human voice) -> speaker;   details after [จอ] -> on-screen panel

Run: python voice_agent.py   then open http://127.0.0.1:7860
"""
import ast
import asyncio
import base64
import datetime
import hashlib
import html
import hmac
import itertools
import json
import logging
import operator
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from decimal import ROUND_HALF_UP, Decimal

import aioice.ice
import httpx
import librosa
import numpy as np
import ollama
import soundfile as sf
import uvicorn
from ddgs import DDGS
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastrtc import AdditionalOutputs, ReplyOnPause, SileroVadOptions, Stream, audio_to_float32, get_current_context
from faster_whisper.vad import VadOptions, get_speech_timestamps
from fastrtc.utils import create_message
from huggingface_hub import hf_hub_download
from pypdf import PdfReader

warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"f5_tts_th.*")
from f5_tts_th.normalize import normalize_text  # noqa: E402
from f5_tts_th.utils_infer import chunk_text, infer_batch_process, load_model as load_f5_model, load_vocoder  # noqa: E402

import config as C  # noqa: E402
import desktop_agent  # noqa: E402
import grounding  # noqa: E402
import google_tools  # noqa: E402
from openthai_asr import OpenThaiASR  # noqa: E402

# Thai log lines on a cp1252 console/pipe on Windows would raise UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("voice")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Thai separates phrases with spaces, not punctuation; "." is not a boundary so "1,250.50" stays whole.
BOUNDARY = re.compile(r"\s")
SENTENCE_END = re.compile(r"(?:ครับ|ค่ะ|คะ|[!?])\s")  # where a spoken sentence ends: one chunk = one tune
SENTENCE_TAIL = re.compile(r"(?:ครับ|ค่ะ|คะ|[!?])\s*$")
DANGLING_HEAD = re.compile(r"^(?:แต่|แล้ว|พร้อม|และ|โดย|ส่วน|หรือ|ซึ่ง|จากนั้น|ต่อมา)")
JUNK = re.compile(r"<think>.*?</think>|[*#_`~>|\[\]{}]|^\s*[-•]\s*|[\u3040-\u30ff\u4e00-\u9fff]", re.S | re.M)
# ponytail: keyword route. Things that change daily: these words force a fresh web_search before the brain speaks.
# a customer's phone read back went to brain.bcsmethai.com as a search (role-play 2026-11-04): such numbers stay
# out of the BC brain query; local docs still get them. ponytail: names still go; skip the prefetch when needed
LONG_NUMBER = re.compile(rf"\d[\d -]{{6,}}\d|(?:(?:{'|'.join(grounding.DIGIT_WORDS)})\s*){{7,}}")  # phone, tax id, account no., in digits or read as words


def masked(text):  # customers' phone/tax/account numbers stay out of the server log
    return LONG_NUMBER.sub("###", text)


# 2026-11-08 06:06-06:09: asked for worked examples, the boss twice made a table without calc and every row was cut.
# ponytail: "ตัวอย่าง" also matches "มีตัวอย่างลูกค้าไหม": one wasted calc line, then the usual answer
EXAMPLE_ASK = re.compile(r"ตัวอย่าง|สมมติ|ตาราง\S{0,4}คำนวณ")
# log 2026-11-09: a plain "50 หาร 0.5 บวก 50" got no calc and ended on CUT_ALL. Checked on masked() words, so a
# phone's dashes are no sum. ponytail: "ลบ 3 รายการ" or a date 10/11 costs one wasted calc round
SUM_ASK = re.compile(r"\d\s*(?:บวก|ลบ|คูณ|หาร)|(?:บวก|ลบ|คูณ|หาร)\s*\d|\d\s*[-+*/x×]\s*\d")
# topics whose worked example genuinely computes numbers (costing, tax, offset, discount); non-math topics (warehouse/location,
# menu workflow, SQL) skip forced round-0 calc (~1.2 s faster) and pick tree/flow/case/SQL examples instead of fake sums
CALC_ASK = re.compile(r"คำนวณ|ต้นทุน|ถัวเฉลี่ย|เฉลี่ย|ไฟโฟ|fifo|(?<!ใบกำกับ)ภาษี|vat|หัก\s*ณ|ส่วนลด|ดอกเบี้ย|ค่าเสื่อม|กำไร|ขาดทุน|"
                      r"ยอดรวม|หักกลบ|มัดจำ|งวด|ผ่อน|ปันส่วน|ราคาขาย|เดบิต|เครดิต", re.I)
FAST_CHAT = re.compile(r"^(?:สวัสดี|หวัดดี|ฮัลโหล|ขอบคุณ(?:มาก)?|ขอคิดดูก่อน|คิดดูก่อน|แค่นี้ก่อน|วางสายก่อน)"
                       r"(?:\s*(?:ครับ|ค่ะ|นะคะ|นะครับ|นะ))*[\s.!?…]*$")
LATEST_ASK = re.compile(r"ข่าว|ล่าสุด|ทอง|น้ำมัน|หุ้น|ค่าเงิน|อัตราแลกเปลี่ยน|ดอลลาร์|บิตคอยน์|คริปโต|อากาศ|ฝนตก|พยากรณ์|"
                        r"ผลบอล|ฟุตบอล|หวย|สลากกินแบ่ง|ราคา.{0,8}วันนี้|วันนี้.{0,8}ราคา")
CONFIRM = re.compile(r"ยืนยัน|กดเลย|ทำเลย|เปิดเลย|ตกลง|ได้เลย|โอเค|ใช่")  # the caller's yes to a held computer action
REFUSE = re.compile(r"ไม่|อย่า|ยกเลิก|หยุด")
# only then does the turn carry hh:mm; no bare ตอนนี้: "ตอนนี้ในกล้องเห็นอะไร" got the clock and answered that the
# camera showed a clock reading 08:54 (the frame was a red circle)
TIME_ASK = re.compile(r"กี่โมง|เวลา|นาที|ทันไหม|สายไหม|ชั่วโมง")
DATE_ASK = re.compile(r"วันที่|วันอะไร|วันไหน|เดือน|ปี|พรุ่งนี้|เมื่อวาน|มะรืน|สัปดาห์|อาทิตย์|นัด|ปฏิทิน|กำหนด")  # ...or the date
TH_DAYS = "จันทร์ อังคาร พุธ พฤหัสบดี ศุกร์ เสาร์ อาทิตย์".split()
TH_MONTHS = "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม".split()


def load_stt():
    return OpenThaiASR(C.STT_MODEL_DIR, lang=C.STT_LANGUAGE, intra_threads=C.STT_THREADS)


def to16k(audio):  # fastrtc frames -> what Whisper and its VAD take
    sr, data = audio
    data = audio_to_float32(np.squeeze(data))
    return data if sr == 16000 else librosa.resample(data, orig_sr=sr, target_sr=16000)


def voiced(audio):
    """The speech only, 16 kHz, by transcribe()'s own VAD: empty = it would transcribe nothing. No padding (its
    0.4 s a side made a 0.5 s word 2 s long), so the length is the speech's own."""
    data = to16k(audio)
    spans = get_speech_timestamps(data, VadOptions(**C.STT_VAD_PARAMS, speech_pad_ms=0))
    return np.concatenate([data[:0], *(data[t["start"] : t["end"]] for t in spans)])


def stt_hears(audio):
    return voiced(audio).size > 0


STT_LOCK = threading.Lock()
LIVE = {}  # handler id -> {"t": monotonic, "fed": samples, "text": last partial, "busy": bool, "stream": OpenThaiStream}


def transcribe(audio):
    return stt.transcribe(to16k(audio))


_DEFAULT_TRANSCRIBE = transcribe


EARLY = {}  # handler -> (the speech so far, its transcription job), started in the silence before a turn ends
EARLY_STT = ThreadPoolExecutor(2)  # ponytail: CTranslate2 runs one GPU job at a time anyway


def early_words(data):
    """What determine_pause already transcribed of this very speech (the caller said nothing after it), else None."""
    flat = data.reshape(-1)
    for key, (head, job) in list(EARLY.items()):
        if head.size <= flat.size and np.array_equal(head, flat[: head.size]) and EARLY.pop(key, None):
            return job.result()
    return None


NUM_WORDS = [*"0123456789", *grounding.DIGIT_WORDS, *grounding.UNIT]
NUM_END = re.compile(rf"(?:{'|'.join(NUM_WORDS)})[\s,.\-]*$")


def inside_number(before, after):
    """A phone number read back in groups ("ศูนย์แปดหนึ่ง หนึ่งสองสาม ...") cut at its spaces became three
    numbers no source had, and the number guard dropped them all. What follows may be only the start of a word
    (DeepSeek streams "ส" then "ี่"), so a number word it could still become counts too."""
    rest = after.lstrip()
    return bool(NUM_END.search(before)) and any(w.startswith(rest[: len(w)]) for w in NUM_WORDS)


def chunk_sentences(tokens, min_chars=20, max_chars=140):
    """Group streamed tokens into speakable chunks: cut at the last sentence end (ครับ, ค่ะ) once min_chars are in,
    or at the last space once max_chars are, never inside a number."""
    # ลุงจืด 2026-11-08 "คุยเป็นธรรมชาติ มนุษย์ มากที่สุด": a cut at any space after 20 characters spoke
    # "เรื่องนี้ต้องบอกตรงๆ / ครับว่าเลือกประเภทต้นทุนเป็น / LOT FIFO ได้", each piece with its own falling tune
    buf = ""
    whole = lambda ms: [x for x in ms if not inside_number(buf[: x.end()], buf[x.end() :])]
    for tok in tokens:
        buf += tok
        if len(buf) >= min_chars and (m := whole(SENTENCE_END.finditer(buf)) or (len(buf) >= max_chars and whole(BOUNDARY.finditer(buf)))):
            cut = m[-1].end()
            if buf[:cut].strip():
                yield buf[:cut]
                buf = buf[cut:]
    if buf.strip():
        yield buf


def until_screen(tokens, screen):
    """Pass spoken tokens through; stop at SCREEN_MARK and put what follows it into `screen`.
    Stopping (not just filtering) lets chunk_sentences flush the last spoken phrase right away."""
    buf, mark = "", C.SCREEN_MARK
    for tok in tokens:
        buf += tok
        if mark in buf:
            speech, rest = buf.split(mark, 1)
            screen.append(rest)
            yield speech
            return
        keep = next((k for k in range(len(mark) - 1, 0, -1) if buf.endswith(mark[:k])), 0)  # maybe a split marker
        yield buf[: len(buf) - keep]
        buf = buf[len(buf) - keep:]
    yield buf


def clean(text):
    return re.sub(r"\s+", " ", JUNK.sub("", text)).strip()


def thai_now(clock=True):
    n = datetime.datetime.now()
    day = f"วัน{TH_DAYS[n.weekday()]}ที่ {n.day} {TH_MONTHS[n.month - 1]} {n.year + 543}"
    return f"{day} เวลา {n:%H:%M} น." if clock else day


def clock_note(heard):
    """The clock for this turn, only when the caller asks the date or time (why: see reply)."""
    bare = re.sub(r"\s+", "", heard)  # the STT spaces words out: "วันนี้ วัน อะไร" missed วันอะไร (call 2026-11-09 06:39)
    if DATE_ASK.search(bare) or TIME_ASK.search(bare):
        return f"\n(ระบบ: นาฬิกาในเครื่องตอนนี้ {thai_now(bool(TIME_ASK.search(bare)))})"
    return ""


_THAI_TERMS = (
    ("BC AI Account", "บีซี เอไอ แอคเคาท์"),
    ("Google Drive", "กูเกิลไดรฟ์"),
    ("Account", "แอคเคาท์"),
    ("FIFO", "ไฟโฟ"),
    ("LIFO", "ไลโฟ"),
    ("LOT", "ล็อต"),
    ("POS", "พีโอเอส"),
    ("VAT", "แวต"),
    ("WHT", "หัก ณ ที่จ่าย"),
    ("SKU", "เอสเคยู"),
    ("SQL", "เอสคิวแอล"),
    ("API", "เอพีไอ"),
    ("ERP", "อีอาร์พี"),
    ("SME", "เอสเอ็มอี"),
    ("PDF", "พีดีเอฟ"),
    ("CSV", "ซีเอสวี"),
    ("Gmail", "จีเมล"),
    ("Google", "กูเกิล"),
    ("Online", "ออนไลน์"),
    ("Cloud", "คลาวด์"),
)
_TERM_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k, _ in _THAI_TERMS) + r")(?![A-Za-z0-9])")
_TERM_MAP = {k.lower(): f" {v} " for k, v in _THAI_TERMS}
_LETTER_TH = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (
    "เอ", "บี", "ซี", "ดี", "อี", "เอฟ", "จี", "เอช", "ไอ", "เจ", "เค", "แอล", "เอ็ม",
    "เอ็น", "โอ", "พี", "คิว", "อาร์", "เอส", "ที", "ยู", "วี", "ดับเบิลยู", "เอ็กซ์", "วาย", "แซด",
)))


def _thai_pronounce(text):
    """Map English business acronyms/terms to spaced Thai phonetics because F5-TTS-THAI's vocab is Thai-only."""
    out = _TERM_RE.sub(lambda m: _TERM_MAP[m.group(1).lower()], str(text))
    out = re.sub(r"(?<![A-Za-z0-9])[A-Z]{2,5}(?![A-Za-z0-9])",
                 lambda m: " " + "".join(_LETTER_TH[c] for c in m.group(0)) + " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _hf_file(repo, filename):
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo, filename)


def load_tts():
    """F5-TTS-THAI v1 + Vocos + preloaded human reference voice tensor kept warm on device."""
    f5_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)
    model = load_f5_model(f5_cfg, _hf_file(C.TTS_MODEL, "model_1000000.pt"), mel_spec_type="vocos",
                          vocab_file=_hf_file(C.TTS_MODEL, "vocab.txt"), device=C.DEVICE)
    vocoder = load_vocoder("vocos", device=C.DEVICE)
    ref_np, sr = sf.read(C.REF_AUDIO_PATH, dtype="float32")
    if ref_np.ndim > 1:
        ref_np = np.mean(ref_np, axis=1, dtype=np.float32)
    if sr != C.TTS_SAMPLE_RATE:
        ref_np = librosa.resample(ref_np, orig_sr=sr, target_sr=C.TTS_SAMPLE_RATE)
    ref_tensor = torch.from_numpy(np.asarray(ref_np, dtype=np.float32)).reshape(1, -1)
    ref_text = C.REF_TEXT.strip()
    if not ref_text.endswith((". ", "。")):
        ref_text += " " if ref_text.endswith(".") else ". "
    return model, vocoder, ref_tensor, ref_text


TTS_LOCK = threading.Lock()  # ponytail: one TTS phrase at a time for every caller; a second model if callers queue


def speak(text):
    model, vocoder, ref_tensor, ref_text = tts
    norm = normalize_text(_thai_pronounce(text)).strip()
    if not norm:
        return np.zeros((1, C.TTS_SAMPLE_RATE // 10), np.float32)
    batches = [b for b in chunk_text(norm, max_chars=120) if b.strip()] or [norm]
    for _ in range(2):
        with TTS_LOCK, torch.inference_mode():
            wav, _, _ = next(infer_batch_process(
                (ref_tensor, C.TTS_SAMPLE_RATE), ref_text, batches, model, vocoder,
                mel_spec_type="vocos", progress=None, nfe_step=C.TTS_STEP, cfg_strength=C.TTS_CFG,
                sway_sampling_coef=-1.0, speed=C.TTS_SPEED, device=C.DEVICE, use_ipa=False,
            ))
        arr = np.asarray(wav, dtype=np.float32).reshape(-1) if wav is not None else np.zeros(0, np.float32)
        if arr.size:
            _, idx = librosa.effects.trim(arr, top_db=40)
            if idx[1] > idx[0]:
                pad_pre, pad_post = int(0.03 * C.TTS_SAMPLE_RATE), int(0.05 * C.TTS_SAMPLE_RATE)
                arr = arr[max(0, int(idx[0]) - pad_pre):min(arr.size, int(idx[1]) + pad_post)]
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak > 0.98:
                arr = arr * (0.95 / peak)
            if arr.size:
                return arr.reshape(1, -1)
    return np.zeros((1, C.TTS_SAMPLE_RATE // 10), np.float32)


def bot_out(text, wav, lead=0.0):  # carries exact audio duration (s); optional silent preroll prevents WebRTC/speaker clipping word 1
    arr = np.asarray(wav, dtype=np.float32).reshape(1, -1)
    pad = int(round(max(0.0, lead) * C.TTS_SAMPLE_RATE))
    out = np.concatenate((np.zeros((1, pad), np.float32), arr), axis=1) if pad else arr
    ev = {"kind": "bot", "text": text, "dur": round(arr.size / C.TTS_SAMPLE_RATE, 2)}
    if pad:
        ev["lead"] = round(pad / C.TTS_SAMPLE_RATE, 2)
    return (C.TTS_SAMPLE_RATE, out), AdditionalOutputs(ev)


# ---------------------------------------------------------------- tools: each returns (cards for the screen, text for the LLM)
def web_search(query):
    hits = DDGS(timeout=5).text(query, region="th-th", max_results=5)
    cards = [{"source": "web", "title": h["title"], "snippet": h["body"], "url": h["href"]} for h in hits]
    text = json.dumps(cards, ensure_ascii=False)
    try:  # full text of the top hit, so the on-screen answer can be complete rather than snippet-deep
        text += "\n\nเนื้อหาจากผลแรก:\n" + str(DDGS(timeout=4).extract(hits[0]["href"], fmt="text_plain")["content"])
    except Exception as e:
        log.info("extract skipped: %s", e)
    return cards, text


def read_doc(path):
    if path.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def search_docs(query):
    # ponytail: re-reads docs/ and scores 1000-char windows by character-trigram overlap (Thai has no spaces);
    # fine for a few hundred pages, switch to bge-m3 embeddings + a cached index when docs/ grows.
    grams = {w[i:i + 3] for w in query.lower().split() for i in range(max(1, len(w) - 2))}
    overlap = lambda s: sum(s.lower().count(g) for g in grams)  # noqa: E731
    hits = []
    for root, _, files in os.walk(C.DOCS_DIR):
        for name in files:
            path = os.path.join(root, name)
            if not name.lower().endswith((".txt", ".md", ".csv", ".pdf")) or os.path.getsize(path) > 20_000_000:
                continue
            try:
                text = read_doc(path)
            except Exception as e:
                log.warning("skip unreadable doc %s: %s", name, e)
                continue
            for i in range(0, max(1, len(text)), 800):
                chunk = text[i:i + 1000]
                if score := overlap(chunk):
                    hits.append((score, os.path.relpath(path, C.DOCS_DIR), chunk.strip()))
    hits.sort(key=lambda h: -h[0])
    hits = hits[:4]
    # the card shows the best-matching line; the LLM gets the whole window around it
    cards = [{"source": "docs", "title": name, "snippet": max(chunk.splitlines(), key=overlap)[:300], "url": None}
             for _, name, chunk in hits]
    found = [{"file": name, "text": chunk} for _, name, chunk in hits]
    return cards, json.dumps(found, ensure_ascii=False) if found else "ไม่พบในเอกสาร"


CALC_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calc_value(node, src):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(ast.get_source_segment(src, node))  # the digits as written: no binary float in a money sum
    if isinstance(node, ast.BinOp) and type(node.op) in CALC_OPS:
        return CALC_OPS[type(node.op)](calc_value(node.left, src), calc_value(node.right, src))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = calc_value(node.operand, src)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("ใช้ได้แค่ตัวเลข + - * / และวงเล็บ")


def calc(lines):
    """ลุงจืด 2026-11-08 "เอา deepseek มาเสริมได้ถ้าไม่รู้หรือทำไม่ได้": a worked example the caller asked for
    (call 05:16, "ช่วยทำตัวอย่างให้หน่อย") had every number cut, none was in a source. Code does the sums, so the
    numbers and labels here are this turn's sources. ponytail: a label's words become sources too, so the boss
    could pass a claim through one; one sum per line, an earlier line's result is written out, not named."""
    out = []
    for line in str(lines).splitlines()[:120]:
        label, _, expr = line.rpartition(":")
        src = expr.replace(",", "").replace("×", "*").replace("÷", "/").strip()[:200]
        if not src:
            continue
        try:
            value = calc_value(ast.parse(src, mode="eval").body, src)
            value = value.quantize(Decimal("0.01"), ROUND_HALF_UP).normalize()  # baht and satang, half up, once
            out.append(f"{label.strip()} {src} = {value:,f}".strip())
        except (SyntaxError, ValueError, ArithmeticError) as e:
            out.append(f"{line.strip()} คิดไม่ได้ ({e})")
    text = "\n".join(out)
    card = {"source": "calc", "title": "เครื่องคิดเลข", "snippet": text[:300], "url": None}
    # eval 2026-11-08: a label without its unit ("ชิ้นละ 100: 10*100") got all 4 spoken example sentences cut over
    # บาท. ponytail: in a turn that used calc, บาท and ชิ้น pass the judge even for another currency; digits are still checked
    return [card], f"{text}\n(หน่วยตามชื่อบรรทัด เช่น บาท ชิ้น)" if text else "ไม่มีสูตรให้คิด"


# ลุงจืด 2026-11-08: what the BC brain and the web don't have, DeepSeek's own knowledge may answer, told to the caller
# as that and never as BC data. The boss's own memory is still cut by the judge: only this worker's answer is a source
GENERAL_ASK = (f"{C.EXPERT}แต่งานนี้ตอบความรู้ทั่วไปได้ทุกเรื่อง ห้ามปฏิเสธว่าอยู่นอกขอบเขต "  # eval r14: the persona made it refuse the sky
               "ตอบคำถามจากความรู้ทั่วไปเป็นภาษาไทย 2-4 ประโยค เฉพาะส่วนที่แน่ใจ ส่วนที่ไม่แน่ใจให้บอกว่าไม่แน่ใจ "
               "ห้ามตอบเรื่องโปรแกรม BC ราคา โปรโมชัน หรือข้อมูลของบริษัทใด ให้ตอบว่าต้องดูจากคลังความรู้ "
               # log 2026-11-08 07:21: it said "ค่าขนส่งถือเป็นค่าจ้างทำของหรือค่าบริการ" (RD ruling 0706/8419: transport is its own class)
               "อัตราภาษี มาตรา หรือข้อกฎหมายไทย ห้ามตอบจากความจำ ให้ตอบว่าในคลังไม่มีเรื่องนี้ ต้องให้นักบัญชียืนยัน "
               "อัตราหรือเรื่องอื่นที่เปลี่ยนตามเวลา ให้บอกด้วยว่าควรเช็กกับแหล่งทางการล่าสุด")


def general_knowledge(question):
    text, _ = worker_chat([{"role": "system", "content": GENERAL_ASK}, {"role": "user", "content": str(question)[:500]}], None)
    text = " ".join(text.split())
    card = {"source": "deepseek", "title": "ความรู้ทั่วไป (ไม่ใช่ข้อมูลของโปรแกรม)", "snippet": text[:300], "url": None}
    return [card], f"ความรู้ทั่วไปจาก DeepSeek ไม่ใช่ข้อมูลของโปรแกรม ต้องบอกผู้ฟังสั้นๆ ว่าเป็นความรู้ทั่วไป: {text}"


class MCPClient:
    """ponytail: minimal MCP streamable-HTTP client (initialize, tools/list, tools/call); one session, no server push."""

    def __init__(self, url):
        self.url, self.http, self.n = url, httpx.Client(timeout=20, limits=httpx.Limits(keepalive_expiry=60)), 0  # kept alive between turns, as DEEPSEEK
        self.headers = {"Accept": "application/json, text/event-stream"}
        self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "thai-voice-agent", "version": "1"}})
        self.http.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=self.headers)

    def rpc(self, method, params):
        self.n += 1
        r = self.http.post(self.url, json={"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}, headers=self.headers)
        r.raise_for_status()
        if sid := r.headers.get("mcp-session-id"):
            self.headers["Mcp-Session-Id"] = sid
        body = r.text
        if "event-stream" in r.headers.get("content-type", ""):
            body = [line[5:] for line in body.splitlines() if line.startswith("data:")][-1]
        msg = json.loads(body)
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message"))
        return msg["result"]

    def read_only_tools(self):  # a voice command must never write into another system
        return [t for t in self.rpc("tools/list", {})["tools"] if (t.get("annotations") or {}).get("readOnlyHint")]

    def call(self, name, args):
        try:
            result = self.rpc("tools/call", {"name": name, "arguments": args})
        except httpx.HTTPStatusError:  # session expired (server restarted) -> new session, retry once
            self.__init__(self.url)
            result = self.rpc("tools/call", {"name": name, "arguments": args})
        return "".join(c.get("text", "") for c in result.get("content", []))


def tool_spec(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def one_arg(name, desc):
    return {"type": "object", "properties": {name: {"type": "string", "description": desc}}, "required": [name]}


TOOLS = {"web_search": web_search, "search_docs": search_docs, "calc": calc,
         "general_knowledge": general_knowledge}
WORKER = {"web_search": "web", "search_docs": "docs", "calc": "calc", "general_knowledge": "deepseek"}  # tool -> team member shown on screen
TOOL_SPECS = [
    tool_spec("web_search", "ค้นข้อมูลจากอินเทอร์เน็ต เช่น ข่าว ราคา สภาพอากาศ หรือข้อเท็จจริงที่ไม่แน่ใจ", one_arg("query", "คำค้น")),
    tool_spec("search_docs", "ค้นเอกสารและไฟล์ของผู้ใช้ในเครื่อง", one_arg("query", "คำสำคัญสั้นๆ คั่นด้วยเว้นวรรค เช่น ร้าน เวลาเปิด")),
    tool_spec("calc", "เครื่องคิดเลข คิดตัวเลขที่ต้องคำนวณก่อนพูดทุกครั้ง เช่น ตัวอย่างที่ผู้ฟังขอ ต้นทุนถัวเฉลี่ย VAT ยอดรวม",
              one_arg("lines", "บรรทัดละหนึ่งสูตร เขียนว่า สิ่งที่คิดพร้อมหน่วย: สูตร ใช้ได้แค่ตัวเลข + - * / และวงเล็บ "
                      "เช่น รับเข้า 10 ชิ้น ชิ้นละ 50 บาท: 10*50 ตัวเลขทุกช่องของตารางรวมถึงยอดคงเหลือและแถวรวมสุดท้ายต้องมีบรรทัดของมัน "
                      "รหัสสมมติ เช่น LOT A001 ให้อยู่ในชื่อบรรทัด")),
    tool_spec("general_knowledge", "ถามความรู้ทั่วไปที่ไม่ใช่เรื่องโปรแกรม BC ราคา หรือของบริษัท ใช้เมื่อคลังความรู้และเว็บไม่มีคำตอบเท่านั้น",
              one_arg("question", "คำถามเต็มประโยค")),
]
STATUS = {"web_search": "กำลังค้นหาในอินเทอร์เน็ต", "search_docs": "กำลังค้นเอกสารในเครื่อง",
          "calc": "กำลังคำนวณ", "general_knowledge": "กำลังถามความรู้ทั่วไป"}
FILLER_TEXT = {"hello": "สวัสดีครับ"}  # pre-rendered greeting played as soon as a call connects
FILLERS = {}  # pre-rendered at warm-up so the hello plays instantly


# ---------------------------------------------------------------- a customer's message for the owner
LEAD_FIELDS = {"name": 80, "company": 120, "phone": 30, "message": 500}  # field -> characters kept
LEAD_LOCK, LEAD_COUNT = threading.Lock(), defaultdict(int)  # call id -> messages it left
LEAD_SPEC = tool_spec(
    # 2026-11-11 08:30: a caller gave a name and a number, the boss searched the brain for them and stored nothing
    "leave_message", f"บันทึกข้อความฝากติดต่อกลับ เรียกทันทีเมื่อลูกค้าสนใจซื้อ ขอให้ติดต่อกลับ หรือฝากเรื่องถึงฝ่ายขายหรือ{C.USER_NAME} "
    "และให้ชื่อกับเบอร์โทรแล้ว ใส่เฉพาะสิ่งที่ลูกค้าพูดเอง ห้ามแต่งเพิ่ม แล้วบอกลูกค้าตามผลที่ได้ "
    "ถ้ายังไม่ได้เรียกหรือเรียกไม่สำเร็จ ห้ามพูดว่าบันทึกหรือส่งเรื่องแล้ว",
    {"type": "object", "required": ["name", "phone"], "properties": {
        "name": {"type": "string", "description": "ชื่อลูกค้าตามที่ลูกค้าบอก"},
        "company": {"type": "string", "description": "ชื่อร้านหรือบริษัท ถ้าลูกค้าบอก"},
        "phone": {"type": "string", "description": "เบอร์โทรติดต่อกลับตามที่ลูกค้าพูด"},
        "message": {"type": "string", "description": "เรื่องที่ลูกค้าอยากให้ติดต่อกลับ สรุปสั้นๆ จากคำของลูกค้า"}}})


def spoken_digits(text):  # "ศูนย์แปด 1-234" -> "081234": a number as the caller gave it, typed or read out
    for i, word in enumerate(grounding.DIGIT_WORDS):
        text = text.replace(word, str(i))
    return re.sub(r"\D", "", text)


def notify_owner():
    """ลุงจืด sees a new message at once: a Windows message box on this PC. No customer detail in it (the screen
    may be shared); the details are at /leads. ponytail: msg.exe only (Windows Pro), anything else just logs."""
    try:
        subprocess.Popen(["msg", "*", "/TIME:86400", f"javis: มีลูกค้าฝากข้อความใหม่ เปิดดูที่ http://127.0.0.1:{C.SERVER_PORT}/leads"],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        log.warning("no message box for the new lead: %s", e)


def leave_message(name="", company="", phone="", message="", heard="", call=""):
    """ลุงจืด 2026-11-11: a caller left a name, a company and a number for a call back and nothing kept them (the
    log masks numbers). One JSON line per message in LEADS_PATH, shown at /leads on the owner line.
    `heard` and `call` come from reply(), not the boss: a number the caller never said is not stored (ห้ามแต่ง).
    ponytail: the name and company are stored as the boss wrote them; a number said in two parts with other
    digits between is refused and asked again; 3 messages per call, one message box per call."""
    lead = {k: " ".join(str(v).split())[:n] for (k, n), v in zip(LEAD_FIELDS.items(), (name, company, phone, message))}
    digits = spoken_digits(lead["phone"])
    if not lead["name"] or not 9 <= len(digits) <= 15:
        return [], "ยังไม่ได้บันทึก ต้องมีชื่อและเบอร์โทรติดต่อกลับ ให้ถามลูกค้าส่วนที่ยังขาด"
    if digits not in spoken_digits(heard):
        return [], "ยังไม่ได้บันทึก เบอร์นี้ไม่ตรงกับที่ลูกค้าพูด ให้ถามเบอร์โทรอีกครั้ง"
    with LEAD_LOCK:
        if LEAD_COUNT[call] >= 3:
            return [], "ยังไม่ได้บันทึกเพิ่ม สายนี้ฝากข้อความครบ 3 ครั้งแล้ว"
        if call not in LEAD_COUNT and len(LEAD_COUNT) >= 512:
            LEAD_COUNT.pop(next(iter(LEAD_COUNT)))
        LEAD_COUNT[call] += 1
        first = LEAD_COUNT[call] == 1
        with open(C.LEADS_PATH, "a", encoding="utf-8") as f:
            at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            f.write(json.dumps({"at": at, "call": call, **lead}, ensure_ascii=False) + "\n")
    if first:
        notify_owner()
    text = " ".join(f"{label} {lead[k]}" for k, label in (("name", "ชื่อ"), ("company", "บริษัท"), ("phone", "เบอร์"),
                                                         ("message", "เรื่อง")) if lead[k])
    card = {"source": "message", "title": "บันทึกข้อมูลติดต่อกลับแล้ว", "snippet": text[:300], "url": None}
    return [card], f"บันทึกข้อความฝากติดต่อกลับแล้ว {text} จะประสานให้ติดต่อกลับ ยังไม่ได้นัดเวลาติดต่อกลับ"


def leads_page():
    """/leads: the messages customers left, newest first. Every field is the caller's text: escaped.
    ponytail: the newest 200, no delete or done mark; edit LEADS_PATH by hand."""
    rows = []
    try:
        with open(C.LEADS_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:  # a line cut short by a crash
                    pass
    except OSError:  # nobody left a message yet
        pass
    cells = []
    for r in reversed(rows[-200:]):
        try:
            at = datetime.datetime.fromisoformat(str(r.get("at"))).astimezone().strftime("%d/%m/%Y %H:%M")
        except ValueError:
            at = ""
        cells.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            at, *(r.get(k, "") for k in LEAD_FIELDS), str(r.get("call", "")).split("_")[0])) + "</tr>")
    return ("<!doctype html><html lang=th><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>ข้อความฝากจากลูกค้า</title><style>body{font:16px/1.5 system-ui,sans-serif;margin:24px;background:#0b1220;color:#e5e7eb}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #334155;padding:8px 12px;text-align:left;vertical-align:baseline}"
            "th{background:#1e293b}</style>"
            f"<h1>ข้อความฝากจากลูกค้า ({len(rows)} รายการ)</h1>"
            + ("<table><tr><th>เวลา</th><th>ชื่อ</th><th>ร้าน/บริษัท</th><th>เบอร์โทร</th><th>เรื่อง</th><th>รหัสสาย</th></tr>"
               + "".join(cells) + "</table>" if cells else "<p>ยังไม่มีลูกค้าฝากข้อความ</p>") + "</html>")


def brain_hits(server, text):
    """search_brain JSON -> only notes that really match, one card each (prefetch and the LLM's own calls alike)."""
    # ponytail: fixed score cut, calibrated on 11 real questions (relevant >= 52, off-topic <= 44 once news/prices
    # go to the web); keyword scores still let some off-topic notes through (POS 71 for a coffee-shop question),
    # so the prompt tells the boss to use only what matches. Upgrade: a reranker.
    try:
        hits = [h for h in json.loads(text) if isinstance(h, dict) and h.get("score", 0) >= C.BRAIN_MIN_SCORE][:3]
    except ValueError:
        return [{"source": server, "title": "search_brain", "snippet": text[:400], "url": None}], text
    cards = [{"source": server, "title": h.get("title", ""), "snippet": str(h.get("snippet", ""))[:300], "url": None, "slug": h.get("slug")} for h in hits]
    found = [{k: h.get(k) for k in ("slug", "title", "snippet")} for h in hits]
    return cards, ("อ่านเต็มด้วย read_note ตาม slug: " + json.dumps(found, ensure_ascii=False)) if hits else "ไม่พบเรื่องที่ตรงกับคำนี้ในคลังความรู้ BC"


NOTE_META = re.compile(r"(?m)^(slug|sources):.*\n?")  # a BC note's metadata lines: its slug and the code paths it came from


def mcp_tool(server, client, name):
    def run(**args):
        text = client.call(name, args)
        if name == "search_brain":
            return brain_hits(server, text)
        if name == "read_note":  # a card the page opens in full: the note's own heading, its text without the metadata
            body = NOTE_META.sub("", text).partition("\n")[2].strip()
            return [{"source": server, "title": text.partition("\n")[0].lstrip("# "), "snippet": body[:300], "url": None, "slug": args.get("slug")}], text
        return [{"source": server, "title": f"{name}: {next(iter(args.values()), '')}", "snippet": text[:400], "url": None}], text
    return run


def load_mcp_tools():
    for server, url in C.MCP_SERVERS.items():
        try:
            client = MCPClient(url)
            tools = client.read_only_tools()
        except Exception as e:
            log.warning("MCP %s unavailable (%s) -> its tools are skipped", server, e)
            continue
        for t in tools:
            schema = {k: v for k, v in t["inputSchema"].items() if k != "$schema"}
            TOOL_SPECS.append(tool_spec(t["name"], t.get("description", ""), schema))
            TOOLS[t["name"]] = mcp_tool(server, client, t["name"])
            WORKER[t["name"]] = server
            STATUS[t["name"]] = f"กำลังค้นคลังความรู้ {server}"
        log.info("MCP %s: %s", server, [t["name"] for t in tools])


def run_tool(name, args, tools):
    try:  # `tools` is the caller's own table: on the public line a tool outside it is just an unknown tool
        cards, text = tools[name](**args)
    except Exception as e:  # unknown tool, bad args, network: tell the LLM instead of crashing the call
        log.warning("tool %s%s failed: %s", name, masked(str(args)), masked(str(e)))  # args may hold a phone read back
        return [], f"เรียก {name} ไม่สำเร็จหรือไม่พบข้อมูล ({e})"
    return cards, text[: C.TOOL_RESULT_CHARS] or "ไม่พบข้อมูล"


# webrtc_id -> what arrived beside the audio: typed chat, a system turn (greeting, reminder).
# ponytail: one small dict per call is never freed; clear it on hang-up if calls ever number in the hundred thousands
PENDING = defaultdict(lambda: {"text": "", "note": "", "carry": (0.0, "")})


def call_id():
    try:
        return get_current_context().webrtc_id
    except RuntimeError:  # no call (warm-up)
        return ""


def tag_call(record):  # every log line names its call ("-" outside one), so watch_calls.py can tell calls apart;
    record.call = call_id().rpartition("_")[2] or "-"  # the call part only: the session part keys a caller's history
    return True


for h in logging.getLogger().handlers:  # lines logged before this point keep the plain format
    h.addFilter(tag_call)
    h.setFormatter(logging.Formatter("%(asctime)s %(call)s %(message)s"))


def take(key, empty):
    p = PENDING[call_id()]
    value, p[key] = p[key], empty
    if len(PENDING) > 256:
        for k in [k for k, v in list(PENDING.items())[:-128] if not v["text"] and not v["note"] and not v["carry"][1]]:
            PENDING.pop(k, None)
    return value


# ---------------------------------------------------------------- the call
SESSIONS = {}  # session id -> that caller's history (user/assistant pairs), loaded from SESSIONS_DIR on first use
SESSION_LOCK = threading.Lock()
SESSION_ID = re.compile(r"[0-9a-f]{8,32}")  # only these name a file: the id comes from the browser


def session(public=False):
    """The caller's own conversation. The browser keeps a session id (localStorage) in front of its webrtc_id
    ("<session>_<call>"), so a new call, a reload or a server restart goes on where the talk left off.
    Customer sessions live under their own "pub-" names: a public caller can never load an owner-line talk."""
    sid = call_id().split("_")[0]
    sid = (f"pub-{sid}" if public else sid) if SESSION_ID.fullmatch(sid) else ""  # "": warm-up, a throwaway history
    with SESSION_LOCK:
        if sid not in SESSIONS:
            if sid and len(SESSIONS) >= 256:
                for old in [k for k in SESSIONS if k][:64]:
                    SESSIONS.pop(old, None)
            try:
                with open(os.path.join(C.SESSIONS_DIR, f"{sid}.json"), encoding="utf-8") as f:
                    SESSIONS[sid] = [m for m in json.load(f) if isinstance(m, dict) and isinstance(m.get("content"), str)]
            except (OSError, ValueError, TypeError):  # new caller, or a file cut short by a crash
                SESSIONS[sid] = []
        return sid, SESSIONS[sid]


def save_session(sid, history):
    if sid:  # write then rename: a crash mid-write keeps the previous file
        path = os.path.join(C.SESSIONS_DIR, f"{sid}.json")
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with SESSION_LOCK:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False)
            os.replace(tmp, path)


# one kept-alive connection: a fresh TLS handshake each call cost ~0.16 s before the first word (measured
# 2026-11-04, 12 s between calls); httpx drops idle ones after 5 s by default, turns come every 10-20 s
DEEPSEEK = httpx.Client(limits=httpx.Limits(keepalive_expiry=60))


# DeepSeek V4 sometimes writes a tool call as its own markup in the text instead of tool_calls (a known leak,
# e.g. github.com/anomalyco/opencode/pull/60575): <｜DSML｜tool_calls>, also with doubled bars or a "calls" wrapper,
# and in the last round with a space after the bars ("<｜｜DSML｜｜ invoke", replay 2026-11-08: 2 of 5 read no call)
DSML_BAR = r"[｜|]+\s*"
DSML_START = re.compile(rf"<{DSML_BAR}DSML")
DSML_INVOKE = re.compile(rf'<{DSML_BAR}DSML{DSML_BAR}invoke\s+name="([^"]+)"\s*>(.*?)</{DSML_BAR}DSML{DSML_BAR}invoke>', re.S)
DSML_PARAM = re.compile(rf'<{DSML_BAR}DSML{DSML_BAR}parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)'
                        rf'</{DSML_BAR}DSML{DSML_BAR}parameter>', re.S)


def dsml_calls(markup):
    """The tool calls in DeepSeek's DSML text, as deepseek_tokens builds them from tool_calls."""
    calls = []
    for n, (name, body) in enumerate(DSML_INVOKE.findall(markup)):
        args = {}
        for key, is_string, value in DSML_PARAM.findall(body):
            try:  # string="false": a JSON value (number, bool, list)
                args[key] = value if is_string != "false" else json.loads(value)
            except ValueError:
                args[key] = value
        calls.append({"id": f"dsml-{n}", "name": name, "args": args})
    return calls


def deepseek_tokens(messages, calls, tools, choice=None):
    """The boss on DeepSeek (OpenAI-style SSE). A tool call arrives in pieces (id and name first, then bits of
    its JSON arguments), so calls are put together when the stream ends."""
    body = {"model": C.DEEPSEEK_MODEL, "messages": messages, "stream": True, "temperature": C.LLM_TEMPERATURE,
            "thinking": {"type": "disabled"}, **({"tools": tools} if tools else {}),  # thinking: +seconds per turn
            **({"tool_choice": choice} if tools and choice else {})}  # "none", or a named tool; needs thinking off
    parts, held, markup = {}, "", ""  # held: a tail that may be the start of DSML; markup: the DSML itself
    with DEEPSEEK.stream("POST", f"{C.DEEPSEEK_URL}/chat/completions", json=body, timeout=C.DEEPSEEK_TIMEOUT,
                      headers={"Authorization": f"Bearer {C.DEEPSEEK_API_KEY}"}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices") or []:
                delta = choice.get("delta") or {}
                for tc in delta.get("tool_calls") or []:
                    p = parts.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                    fn = tc.get("function") or {}
                    p["id"] += tc.get("id") or ""
                    p["name"] += fn.get("name") or ""
                    p["args"] += fn.get("arguments") or ""
                if markup:  # past the start of the markup: no more words for the caller
                    markup += delta.get("content") or ""
                    continue
                held += delta.get("content") or ""
                if m := DSML_START.search(held):
                    yield held[: m.start()]
                    held, markup = "", held[m.start() :]
                    continue
                cut = held.rfind("<")  # ponytail: a "<" waits up to 12 characters to see if DSML follows
                cut = len(held) if cut == -1 or len(held) - cut > 12 else cut
                yield held[:cut]
                held = held[cut:]
    yield held
    if markup and not parts and tools:  # with no tools offered there is nothing to run: the markup just goes
        log.info("DeepSeek wrote its tool call as text: %s", ", ".join(c["name"] for c in dsml_calls(markup)) or "none whole")
        calls.extend(dsml_calls(markup))
    for p in parts.values():
        try:
            args = json.loads(p["args"] or "{}")
        except ValueError:  # broken JSON: run_tool then tells the boss the call failed
            args = {}
        calls.append({"id": p["id"], "name": p["name"], "args": args if isinstance(args, dict) else {}})


def llm_tokens(messages, calls, tools, choice=None):
    """The boss on DeepSeek. If it fails before saying anything (network, quota, key) the caller hears BRAIN_DOWN,
    never a guess; half an answer already out just ends the turn."""
    spoke = False
    try:
        for tok in deepseek_tokens(messages, calls, tools, choice):
            spoke = spoke or bool(tok)
            yield tok
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("DeepSeek failed (%s %s)", type(e).__name__, getattr(getattr(e, "response", None), "status_code", ""))
        calls.clear()
        if not spoke:
            yield BRAIN_DOWN


GEO, GEO_LOCK, GEO_LAST = {}, threading.Lock(), [0.0]  # place -> "lat,lon" (found ones only)
ROW_NO = re.compile(r"\s*\|\s*(\d{1,3})\s*\|")  # a table row's first cell, a plain count
MAP_LINE = re.compile(r"^\s*\[แผนที่\]\s*(.+?)\s*$")


def geocode(place):
    """Coordinates of a place the boss put on screen, from OpenStreetMap Nominatim; None if not found or offline.
    Their policy: at most 1 request a second for the whole app, an identifying User-Agent, cache the results."""
    if place not in GEO:
        with GEO_LOCK:  # ponytail: one global lock = the 1 req/s limit; a local Nominatim if maps get busy
            time.sleep(max(0.0, GEO_LAST[0] + 1 - time.monotonic()))
            try:
                r = httpx.get("https://nominatim.openstreetmap.org/search", timeout=5, headers={"User-Agent": C.GEO_USER_AGENT},
                              params={"q": place, "format": "jsonv2", "limit": 1, "accept-language": "th"})
                hit = r.json()[:1] if r.status_code == 200 else []
            except (httpx.HTTPError, ValueError):
                hit = []
            GEO_LAST[0] = time.monotonic()
        if hit:
            GEO[place] = f'{float(hit[0]["lat"]):.5f},{float(hit[0]["lon"]):.5f}'
    return GEO.get(place)


# SQL words a query may use that no note spells out; any other name in a code line must be in the sources
CODE_WORDS = set("select from where and or not in is null as on join left right inner outer full cross group by order "
                 "having sum count min max avg round abs coalesce nullif case when then else end between like ilike "
                 "distinct limit offset union all with asc desc cast numeric decimal integer int bigint smallint text "
                 "varchar char boolean date timestamp timestamptz now current_date current_timestamp interval filter "
                 "true false exists any lower upper trim extract over partition row_number rank dense_rank lag lead "
                 "greatest least concat replace substring split_part to_char to_date date_trunc string_agg array_agg "
                 "json jsonb jsonb_agg unnest generate_series lateral using values".split())
SQL_TYPES = re.compile(r"(?i)\b(?:numeric|decimal|varchar|char|character\s+varying|timestamp\w*)\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\)")
SQL_OWN = re.compile(r"(?i)(?:\bas\s+([A-Za-z_]\w*)|\bwith\s+([A-Za-z_]\w*)\s+as\b|,\s*([A-Za-z_]\w*)\s+as\s*\()")


def code_unsourced(line, known, sources, own_names=None):
    """What a code line makes up: a table, column or number no source has. CTE/column aliases declared with AS
    carry across the code block (`own_names`), SQL type precisions like numeric(18,2) are stripped before number
    checking, and a source range "54-73" covers the flags between."""
    low = sources.lower()
    declared = {w.lower() for g in SQL_OWN.findall(line) for w in g if w} - CODE_WORDS
    if own_names is not None:
        own_names |= declared
    allowed = own_names if own_names is not None else declared
    names = {w for w in re.findall(r"[A-Za-z_]\w*", line)
             if len(w) > 2 and w.lower() not in CODE_WORDS and w.lower() not in low and w.lower() not in allowed}
    spans = {str(n) for a, b in re.findall(r"\b(\d{1,4})\s*-\s*(\d{1,4})\b", sources)
             if int(a) < int(b) <= int(a) + 100 for n in range(int(a), int(b) + 1)}
    clean_sql = SQL_TYPES.sub(" ", line)
    return names | grounding.unsourced(re.sub(r"[A-Za-z_]\w*|,", " | ", clean_sql), set(known) | spans)


def panel_events(tokens, screen, known, sources, turn):
    """The [จอ] part a line at a time: a line shows once complete, its numbers sourced and the judge backing it.
    Progress lives in `turn`, so a later tool round goes on from the lines already checked."""
    kept = turn.setdefault("kept", [])
    for tok in itertools.chain(tokens, ["\n"]):  # the extra "\n" closes the last line
        screen.append(tok)
        lines = "".join(screen).split("\n")[:-1]
        if len(lines) <= turn.get("lines", 0):
            continue
        for ln in lines[turn.get("lines", 0):]:
            # a table's first row is its column names, no claim: the judge cut "| ขั้น | สิ่งที่ระบบทำ | สูตร/เงื่อนไข |"
            # (2026-11-08 04:59) and the page, with no head row, ran the whole table into one paragraph
            head = ln.lstrip().startswith("|") and not (kept and kept[-1].lstrip().startswith("|"))
            # a table's own row numbers are no claim: rows 10, 11, 13 of a 13-step table were cut (call 2026-11-09
            # 06:38), 1-9 pass as lone digits. A first cell that counts on from the row above is that count.
            # ponytail: tables only; a numbered list past 9 items still has its 10. checked
            row = not head and ROW_NO.match(ln)
            if head or not ln.lstrip().startswith("|"):
                turn["row_no"] = 0
            facts = "|" + ln[row.end():] if row and int(row[1]) == turn.get("row_no", 0) + 1 else ln
            if row:
                turn["row_no"] = int(row[1])
            if fence := ln.lstrip().startswith("```"):  # only ```sql is checked as SQL code; plain fences still go through judge
                turn["code"] = False if turn.get("code") else ln.lstrip()[3:].strip().lower().startswith("sql")
                turn["sql_own"] = set()
            # a code line is checked by code: the judge cut true SQL lines one by one and left a broken query (eval 2026-11-08)
            code = turn.get("code") and not fence
            if fence or not re.search(r"\w", ln) or (not (bad := code_unsourced(ln, known, sources, turn.setdefault("sql_own", set())) if code else grounding.unsourced(facts, known))
                                                    and (head or code or is_label(ln) or judge(sources, ln, "\n".join(kept)))):
                m = MAP_LINE.match(ln)  # the page draws a map where the line carries "@lat,lon"
                kept.append(f"{ln} @{geo}" if m and (geo := geocode(m[1])) else ln)  # blank and |---| lines need no check
            else:  # logged like a spoken cut: javis 2026-11-07 showed a panel of only CUT_SHOW, with no trace why
                log.info("cut a panel line (%s): %s", f"{'numbers or names' if code else 'numbers'} {masked(' '.join(sorted(bad)))}" if bad else "judge", masked(ln))
                turn["panel_cut"] = True
        turn["lines"] = len(lines)
        if turn.get("panel", "") != (text := "\n".join(kept + [grounding.CUT_SHOW] * turn.get("panel_cut", 0)).strip()):
            turn["panel"] = text
            yield AdditionalOutputs({"kind": "panel", "text": text})


BASE_PROMPT = C.SYSTEM_PROMPT  # load_workers appends what is true of this setup (team, Google, desktop) after it
# how it names a source, and the small talk a greeting may have ("เฮ้ ลุงจืด วันนี้สบายดีไหม" was cut as a new fact)
SAY_WORDS = (f"{C.USER_NAME} จากคลังความรู้บีซี จากเว็บ จากเอกสารในเครื่อง ไฟล์ แหล่งข้อมูล ลูกทีม สบายดี "
             "พนักงาน ตอบคำถาม คำถาม ผู้ใช้งาน ดูแล ประสาน ฝ่ายขาย เลขา รับสาย โทร ติดต่อกลับ แจ้ง รายละเอียด ชื่อ เบอร์ ช่องทางติดต่อ ร้าน บริษัท ธุรกิจ ทำบัญชี ความต้องการ สนใจ "
             "ราคา แพ็กเกจ โปรโมชัน "  # what the persona says when the brain has no price: "ให้ลุงจืดติดต่อกลับไปแจ้งราคา"
             "นักบัญชี ยืนยัน "  # config.py's own tax/accounting disclaimer: "ต้องให้ลุงจืดหรือนักบัญชียืนยัน"
             "ขั้นตอน แผนที่ การ์ด เคส กรณีตัวอย่าง "  # the panel's [ขั้นตอน], [แผนที่], [การ์ด] and > casebox marks
             # pronouns and polite talk: the judge still called ผม / เดี๋ยว new_info now and then and cut a whole line
             "ผม ดิฉัน คุณ ครับ ค่ะ คะ ยินดี ขอบคุณ ขอโทษ เดี๋ยว ขอ ถาม ทราบ รับทราบ เวลา ช่วย ตอนนี้ ก่อน "
             # log 2026-11-08 12:27 "รับสายแทนลุงจืดอยู่ครับ" (แทน, also 09:17) and the honest "ยังไม่แน่ใจครับ" (แน่ใจ)
             # were cut as new_info (judge replay 2 of 2 each)
             # log 2026-11-08 13:29 "มีอะไรให้ช่วยเรื่องโปรแกรมบัญชีไหมครับ" was cut over the question word ไหม (judge replay 2 of 2);
             # รึเปล่า is the one other question word still left as new (มั้ย หรือเปล่า หรือยัง บ้าง นะ are already filtered)
             "แทน แน่ใจ ไหม รึเปล่า ค้น ดู ตรวจสอบ สักครู่ "
             # what it sells and asks about (the prompt's sales lines are no source): the judge cut "ผมช่วยเรื่อง
             # โปรแกรมบีซีเอไอแอคเคาท์ครับ" over โปรแกรม, and a customer's "ช่วยอะไรได้บ้าง" got silence
             "โปรแกรมบีซีเอไอแอคเคาท์ บีซี เอไอ แอคเคาท์ BC Ai Account การใช้งาน")
# the judge sorts only the words code found in no source: asked "is this sentence backed?" the 8B model said yes to
# almost anything (weather, free parking, a made-up email), asked word by word it caught them (20/23 in a replay)
def is_label(ln):
    """The first line of a made-up example (ลุงจืด 2026-11-09 "ต้องแสดงตัวอย่างด้วยทุกครั้ง"): no claim, no judge.
    Not in SAY_WORDS: in the judge's sources it cut small talk, 4/5 -> 3/5 passed every run (eval 2026-11-09)."""
    # its words, not its exact text: the boss wrote "### ตัวอย่างสมมติ (ไม่ใช่ข้อมูลจริง)" (probe 2026-11-09)
    return C.SAMPLE_LABEL.split()[0] in ln and not grounding.new_words(ln, C.SAMPLE_LABEL)


def judge_format(new):  # one entry per new word, those words only: shorter output, none skipped or added
    entry = {"type": "object", "required": ["word", "type", "source_word"], "properties": {
        "word": {"type": "string", "enum": new}, "type": {"type": "string", "enum": ["talk", "same_as_source", "new_info"]},
        "source_word": {"type": "string"}}}
    return {"type": "object", "required": ["words"], "properties": {
        "words": {"type": "array", "items": entry, "minItems": len(new), "maxItems": len(new)}}}


JUDGE_ASK = (
    "SOURCES:\n{src}\n\nA Thai voice assistant already said: {before}\nIt now wants to say: {piece}\n"
    "Words of it that appear nowhere in SOURCES: {new}\n\n"
    # ลุงจืด 2026-11-04 "ให้คุยเหมือนเพื่อน": a question about the caller's own side is talk (the judge cut
    # "สองสาขานี่สต็อกแยกกันหรือใช้ร่วมกันครับ"), a guess or offer is not. Eval on recorded calls: 2 of 19 wrong, was 5
    "Give each of those words a type. talk = a word about the conversation itself (a greeting, polite word, "
    "pronoun, asking, meaning, helping, finding, knowing, checking, being connected, or saying that details/a table/a "
    "query are shown on screen: ผม, หมายถึง, เช็ก, ไม่พบ, เขียนไว้บนจอ, ดูตารางบนจอ), "
    # log 2026-11-08 08:59 "สนใจเรื่องไหนเป็นพิเศษครับ", "อยากให้ช่วยเรื่องไหนเป็นพิเศษ" cut; replay 09:04 cut "…เป็นหลักครับ" on เป็นหลัก
    "a word that only says which part matters more (เป็นพิเศษ, เป็นหลัก), "
    "or a word the assistant uses to ask the user about the user's own side (their shop, work, way of doing "
    "things, stock, branches, bills, problem, wish or feeling) or to react kindly to what the user just said "
    "(เหนื่อยไหม, สาขาไหน, ติดปัญหาตรงไหน, สต็อกแยกกันหรือใช้ร่วมกัน, ช้าตอนพิมพ์, อยากให้ช่วยคำนวณให้เองใช่ไหม, "
    "เดินทางลำบาก, ยินดีที่ได้รู้จัก, ไม่ต้องรีบ), "
    # log 2026-11-08 10:12, 10:33: "ในคลังไม่มีตัวเลขจำนวนลูกค้าครับ" was cut and the caller heard CUT_ALL instead;
    # saying what is not there is the honest answer. A check of 6 lines (2 such, 4 made-up claims): 5 right, was 3
    "or a word naming what SOURCES do not have, in a sentence that only says SOURCES or the คลัง have no such "
    "thing and claims nothing else (ในคลังไม่มีตัวเลขจำนวนลูกค้า, ยังไม่มีคิวรี่สำเร็จรูปในคลัง), "
    "or a mock code, mock name or scenario word inside a hypothetical sample (such as WH01, A01, SKU01, BK001, INV001, "
    "คลังหน้าร้าน, ชั้นวาง, โซน, บิลขาย, ชื่อลูกค้าผิด, คู่ค้า, ร้านตัวอย่าง) when 'already said' contains 'ตัวอย่างสมมติ' "
    "and the line claims no new product feature, price, tax rate or rule. "
    "same_as_source = a Thai reading, other spelling or plain "
    "synonym of a word that IS in SOURCES; put that SOURCES word in source_word (แฟล็ก -> flag). "
    # log 2026-11-07 22:02 replays: true panel lines cut for "(Ctrl+S)", "(IA)", "**2)" though the note has Ctrl+S, IA
    "A SOURCES word with brackets, stars or other marks stuck to it is same_as_source too ((Ctrl+S) -> Ctrl+S). new_info = "
    "anything the assistant states or offers about itself, its product, a service or the world that SOURCES do "
    "not say, even inside a question, and any guess or advice it states (not asks) about the user, their shop or "
    "their customers (น่าจะชอบ, ต้องระวัง): a feature, service, price, free thing, discount, trial, visit, "
    "promise, place, day, weather, person, team, event, step or condition (ส่งฟรี, ไปติดตั้งให้, ทดลองใช้ฟรี, "
    "รองรับหลายสาขา, วันเสาร์, ต้องเซ็นชื่อ). "
    # eval 2026-11-08: "แถวบ้านผมก็ตกหนักเหมือนกันครับ" passed as talk; the assistant has no home or life
    "A life of its own is new_info too: its home, family, the weather where it is, what it did or felt "
    "(แถวบ้านผม, ที่บ้านผม, ผมก็เคย). source_word is \"\" unless same_as_source. Answer JSON."
)
# said only when every phrase of an answer was cut. ลุงจืด 2026-11-08 "ผมไม่เจอในแหล่งข้อมูล ขอไม่เดานะครับ อันนี้พูดบ่อยไป":
# an answer with some phrases kept now just ends on them (the panel still notes the cut), and this asks for more words
CUT_ALL = "เรื่องนี้ผมยังหาข้อมูลที่แน่ใจไม่เจอครับ ช่วยเล่าเพิ่มอีกนิดได้ไหมครับ"
BARE_ACK = re.compile(r"(?:ได้|ได้เลย|ครับ|ค่ะ|ครับผม|ใช่|เข้าใจ|รับทราบ|โอเค|อ๋อ)(?:\s*(?:ครับ|ค่ะ|เลยครับ|นะครับ))?")
SCREEN_ONLY = "รายละเอียดผมแสดงไว้บนจอแล้วครับ"
# DeepSeek at times ends a round on a promise ("ขอผมอ่านรายละเอียดก่อนนะครับ") with no tool call, and the turn went
# quiet (javis 2026-11-07: 5 turns of one call, the first with no history). ponytail: a phrase list; a new wording
# slips past until it is added here
STALL = re.compile(r"(ขอ|เดี๋ยว)(ผม)?(ให้(ลูก)?ทีม)?(อ่าน|เปิด|ค้น|ดู|เช็ก|ตรวจ)")
STALL_NUDGE = "(ระบบ: คุณบอกว่าจะอ่านหรือค้นเพิ่ม แต่ยังไม่ได้สั่งลูกทีม ให้เรียกเครื่องมือตอนนี้เลย หรือตอบจากผลที่ลูกทีมค้นให้แล้ว ห้ามบอกว่าจะอ่านอีก)"
LAST_ROUND = {"role": "user", "content": "(ระบบ: ค้นครบแล้ว ไม่มีเครื่องมือให้เรียกอีก ตอบผู้โทรจากผลที่ลูกทีมค้นให้แล้วตอนนี้เลย "
                                         "ถ้าไม่มีส่วนที่ตรง ให้บอกสั้นๆ ว่ายังไม่มีข้อมูลเรื่องนี้)"}
def plain_turn(messages):
    """The last round's view of the turn: tool calls dropped and their results put in the user's words, so DeepSeek has
    no call to copy. With the calls kept it wrote one more as DSML text, offered tools or not (tool_choice "none"
    too), and with nothing to run the turn ended on CUT_ALL: replay 2026-11-08, 3 of 10 POS asks."""
    out = []
    for m in messages:
        if m["role"] == "tool":
            m = {"role": "user", "content": f"(ผลที่ลูกทีมค้นให้: {m['content']})"}
        elif m.get("tool_calls"):
            if not m["content"]:  # a bare call says nothing: the results follow the question itself
                continue
            m = {"role": "assistant", "content": m["content"]}
        if out and out[-1]["role"] == m["role"] == "user":  # one user message, as the chat template expects
            m = {"role": "user", "content": f"{out.pop()['content']}\n{m['content']}"}
        out.append(m)
    return out


BRAIN_DOWN = "ขอโทษครับ ตอนนี้ระบบตอบไม่ได้ชั่วคราว รบกวนถามใหม่อีกครั้งนะครับ"
# the secretary's own fixed lines: words from its greeting or status talk are no new claim
OWN_WORDS = "\n".join([SAY_WORDS, BRAIN_DOWN, *FILLER_TEXT.values()])
# talk about the talk itself (log 2026-11-09..10: 44 sentences cut by the judge, half of them "ช่วยสะกดหรือบอกอีกนิด
# ได้ไหมครับ", "ขอโทษที่เข้าใจคลาดไป", "ได้ยินชัดครับ"): no product, offer, person, place or time word may go in here.
# These shrink the new-word list only: in the judge's SOURCES they flipped small talk (A/B 2026-11-10: 3/5 -> 2/5)
TALK_WORDS = ("หมายถึง ความหมาย เข้าใจ คลาด อธิบาย สะกด อีกครั้ง ซ้ำ เล่าเพิ่ม อีกนิด สั้นๆ ได้ยิน ชัด อ๋อ งั้น ไม่เจอ ไม่พบ ไม่มีข้อมูล รู้เรื่อง "
              "เขียน บน จอ หน้าจอ ตาราง แสดง สรุป เปรียบเทียบ เทียบ ไล่ ทีละ บรรทัด แถว หัวข้อ รายการ ตรงๆ โดยตรง เฉพาะ ทั้งก้อน")
ABOUT_ASK = "\nถ้าถูกถามว่าเป็นใครหรือทำอะไรได้ ให้ตอบว่า "  # the public prompt's last line: its self-description


def setup_facts(prompt):
    """What load_workers said is true of this line, for the judge's sources: the who-line stays out (as a source it
    flipped small talk 3/5 -> 2/5, A/B 2026-11-10); judge() uses it for the new-word list only."""
    return prompt.removeprefix(BASE_PROMPT).partition(ABOUT_ASK)[0]


def judge(sources, piece, before):
    """ห้ามแต่งเรื่อง: True only if `piece` states nothing the sources lack (numbers have their own check).
    The prompt alone did not hold: the boss added "ต้องระวังเรื่องการจัดส่งและรับชำระ" that no result had."""
    # TALK_WORDS and the self-description pass words unjudged but are no SOURCES text (A/B 2026-11-10: small talk 4/5,
    # must-cut 10/10). ponytail: the public self-description on the owner line too; its words are all true there
    if not (new := grounding.new_words(piece, f"{sources}\n{before}\n{TALK_WORDS}\n{PUBLIC['prompt'].partition(ABOUT_ASK)[2]}")):
        return True  # every word is a source word: no model call, so most phrases cost nothing
    low = sources.lower()  # a claimed source word must really be there (code checks the model)

    def verdict(src):
        # 13 JSONDecodeErrors in the call log, 4 of 8 eval runs (run 7 cut a true ถัวเฉลี่ย line). Cause (spy 2026-11-10,
        # 4 of 4): done_reason=length, prompt 4061-4091 of the judge's 4096-token context, 5-35 tokens left for the JSON;
        # 6000 chars of dense Thai (JUDGE_MAX_CHARS) can fill it, and the same prompt again broke the same way 4/4.
        # So the retry reads the nearer half of the source. ponytail: the far half is unread on the retry (a would-be
        # cut either way); the upgrade is a smaller JUDGE_MAX_CHARS or a larger num_ctx, each with a judge eval before/after
        tail = (f"{C.SAMPLE_LABEL}\n" if C.SAMPLE_LABEL.split()[0] in before and C.SAMPLE_LABEL.split()[0] not in before[-600:] else "") + (before[-600:] or "-")
        for attempt in (1, 2):
            ask = JUDGE_ASK.format(src=src if attempt == 1 else src[len(src) // 2:], before=tail,
                                   piece=piece, new=", ".join(new))
            r = None
            try:  # no num_ctx: the same options as warm_up, or Ollama reloads the model (~25 s)
                r = llm.chat(model=C.JUDGE_MODEL, messages=[{"role": "user", "content": ask}], format=judge_format(new),
                             think=False, keep_alive=-1, options={"temperature": 0})
                words = json.loads(r.message.content)["words"]
                break
            except Exception as e:  # no verdict is no check: "ไม่เจอ" beats a guess
                out = getattr(getattr(r, "message", None), "content", None)
                log.warning("judge failed (%s) try %d/2, done_reason=%s prompt_eval_count=%s content=%r", type(e).__name__,
                            attempt, getattr(r, "done_reason", None), getattr(r, "prompt_eval_count", None),
                            masked(out[:80]) if out else None)
        else:
            return False  # both tries failed: treating the piece as unsourced
        return bool(words) and all(w["type"] == "talk" or (w["source_word"].strip() and w["source_word"].strip().lower() in low)
                                   for w in words)
    # a long turn's right note can sit before the tail: a would-be cut gets one more look at the source lines nearest
    # the piece. ponytail: cut candidates only, so the common path keeps Ollama's cached prefix (one more judge call per cut)
    return verdict(sources[-C.JUDGE_MAX_CHARS:]) or (
        len(sources) > C.JUDGE_MAX_CHARS and verdict(grounding.near_sources(piece, sources, C.JUDGE_MAX_CHARS)))


def worker_chat(messages, tools):
    """A worker's own brain call, on DeepSeek like the boss: (text, calls)."""
    calls = []
    return "".join(deepseek_tokens(messages, calls, tools)), calls


# ลุงจืด 2026-11-08 "ต้องจัดคำถามให้เหมาะก่อนค้นหา เพราะบางครั้งก็สะกดผิด หรือคนถามสับสน": the caller's raw words found
# little (STT wrote FIFO as "ไฟโฟ"; "การคำนวณต้นทุนล่ะ" glued to the question before it found depreciation notes)
# first wording (2026-11-08 check) dragged the last topic into a new question and added terms not asked (FIFO to a lot question)
SEARCH_ASK = (f"{C.EXPERT}เขียนคำค้นคลังความรู้โปรแกรมบัญชี BC สำหรับคำพูดล่าสุดของผู้โทร บรรทัดแรกเป็นคำค้น 2-6 คำ (ถ้าคำถามมีหลายแง่มุมในเรื่องเดียวกัน คั่นคำค้นเพิ่มอีก 1-2 มุมด้วยเครื่องหมาย ; เช่น ลูกหนี้เจ้าหนี้คนเดียวกัน หักกลบ ; รายงานสถานะลูกหนี้เจ้าหนี้) ไม่ต้องอธิบาย "
              "แก้คำที่ระบบฟังเสียงเขียนผิดหรือสะกดผิด (ไฟโฟ -> FIFO, พีโอเอส -> POS, หลอด หรือ รถ ในเรื่องสต็อก -> LOT ล็อต, คิวรี่ คิวลี่ หรือ Q -> query SQL) คำที่คนไทยพูดทับศัพท์ให้ใส่ทั้งคำไทยและคำอังกฤษ "
              "ถ้าคำพูดล่าสุดยังถามต่อในเรื่องเดิมหรือพูดคำสั้นๆ ที่อยู่ในหัวข้อที่กำลังคุยได้ (เช่น คุยเรื่องต้นทุนสินค้าอยู่แล้วถามว่า \"FIFO?\", \"แล้วยกเลิกล่ะ\", \"30 บรรทัด\", \"ขอ query\") ให้คงหัวข้อเรื่องที่กำลังคุยอยู่ไว้ในคำค้นด้วย "
              "ถ้าเปลี่ยนไปถามระบบหรือเรื่องอื่นชัดเจน ใช้เรื่องใหม่อย่างเดียว ห้ามเอาเรื่องก่อนมาปน และห้ามใส่คำอื่นที่ผู้โทรไม่ได้พูดถึง "
              "ถ้าคำพูดล่าสุดถามหรือขอเรื่องอื่นที่ไม่ใช่เรื่องที่คุยอยู่ (ระบบอื่นของโปรแกรมเดียวกันก็นับเป็นเรื่องอื่น เช่น คุยเรื่องต้นทุนสินค้าอยู่แล้วถามเรื่องระบบเงินเดือนพนักงาน) "
              "และผู้โทรไม่ได้พูดเองว่าขอเปลี่ยนเรื่อง (อาจเพราะระบบฟังเสียงผิด) ต้องเพิ่มบรรทัดที่สองว่า เปลี่ยนเรื่อง: เรื่องที่คุยอยู่ | เรื่องใหม่ที่ได้ยิน "
              "ไม่ต้องเพิ่มเมื่อเป็นคำทัก คุยเล่น ขอบคุณ คำตอบที่เลขาถาม หรือคำถามต่อในเรื่องเดิม")


# the topic ask is a fixed line (reply), so the next turn reads its two topics back
TOPIC_ASK = re.compile(r"จะเปลี่ยนไปคุยเรื่อง(.+?) หรือยังคุยเรื่อง(.+?)ต่อครับ")


def swarm_queries(query, heard, history):
    """Split a Swarm planner line ('q1 | q2 | q3') into distinct parallel search angles (up to 3)."""
    if query:
        parts = [" ".join(p.split())[:120] for p in re.split(r"[;|]", query)]
        if out := list(dict.fromkeys(p for p in parts if p))[:3]:
            return out
    last = next((m["content"].splitlines()[0] for m in reversed(history) if m["role"] == "user" and m["content"]), "")
    raw = f"{last} {heard}" if len(heard) < 20 and last and not last.startswith("(") else heard
    return [" ".join([raw, *grounding.acronyms(raw)]).strip()[:150]]


def search_words(history, heard):
    """The caller's words made into a search: misheard words fixed, Thai and English terms, a follow-up made whole.
    Returns (words, jump): jump is (the topic so far, the new one heard) when the words leave the topic unasked.
    ponytail: one more DeepSeek call before the first word; "" on failure, and the raw words are searched"""
    talk = [f"{'ผู้โทร' if m['role'] == 'user' else 'เลขา'}: {m['content'].partition(C.SCREEN_MARK)[0].strip()[:150]}"
            for m in history[-4:] if m["content"] and not m["content"].startswith(("(", C.SILENT_MARK))]
    try:
        text, _ = worker_chat([{"role": "system", "content": SEARCH_ASK},
                               {"role": "user", "content": "\n".join([*talk, f"ผู้โทร (ล่าสุด): {heard}"])}], None)
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("search words failed (%s)", type(e).__name__)
        return "", None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # real calls (log 2026-11-09): the jump line came first, or a reply sentence came instead of a query, and that
    # was searched. A line that ends like speech is no query: "" and the raw words are searched
    first = next((line for line in lines if not line.startswith("เปลี่ยนเรื่อง")), "")
    first = "" if re.search(r"(?:ครับ|ค่ะ|คะ)[\s.!?…]*$", first) else first
    jump = next(([t.strip() for t in line.partition(":")[2].split("|")] for line in lines if line.startswith("เปลี่ยนเรื่อง")), [])
    # a first question has no topic to leave
    jump = tuple(jump) if len(jump) == 2 and all(jump) and any(t.startswith("ผู้โทร:") for t in talk) else None
    return " ".join(first.split())[:200], jump


def load_workers():
    """Workers beyond search/docs/bcbrain: Google (once google_login.py has run) and the computer."""
    if google_tools.connected():
        TOOLS.update(google_tools.TOOLS)
        TOOL_SPECS.extend(google_tools.TOOL_SPECS)
        STATUS.update(google_tools.STATUS)
        WORKER.update({n: n.split("_")[0] for n in google_tools.TOOLS})  # gmail_*, calendar_*, drive_*
        C.SYSTEM_PROMPT += "\nอีเมล นัดหมาย ไฟล์ใน Drive ใช้เครื่องมือ gmail calendar drive "
    else:
        log.info("ยังไม่ได้เชื่อม Google (รัน google_login.py) -> ข้าม Gmail/Calendar/Drive")
        # set before warm_up, so the cached prompt prefix stays the same for every call
        C.SYSTEM_PROMPT += "\nตอนนี้ยังไม่ได้เชื่อม Gmail ปฏิทิน และ Google Drive ถ้าถูกถามเรื่องเหล่านี้ให้บอกว่ายังไม่ได้เชื่อม"
    if os.path.exists(desktop_agent.AD):
        TOOLS["desktop_task"] = lambda task: desktop_agent.desktop_task(task, worker_chat)
        TOOL_SPECS.append(tool_spec("desktop_task", "ส่งงานให้ลูกทีมดูหรือควบคุมคอมพิวเตอร์ เช่น ดูว่าเปิดโปรแกรมอะไรอยู่ "
                                    "อ่านข้อความในหน้าต่าง เปิดโปรแกรม กดปุ่ม (พิมพ์ข้อความไม่ได้)",
                                    one_arg("task", "งานที่ต้องทำบนคอมพิวเตอร์ เขียนให้ชัด")))
        STATUS["desktop_task"] = "ลูกทีมกำลังดูคอมพิวเตอร์"
        WORKER["desktop_task"] = "desktop"
        C.SYSTEM_PROMPT += "\nงานบนหน้าจอคอมพิวเตอร์ เช่น ดูว่าเปิดโปรแกรมอะไรอยู่ อ่านหน้าต่าง เปิดโปรแกรม กดปุ่ม ให้ส่ง desktop_task ถ้าลูกทีมรายงานว่ารอยืนยัน ให้บอกสั้นๆ ว่าจะทำอะไร แล้วถามว่ายืนยันไหม ให้พูดว่ายืนยัน "
    # the boss once listed an email worker that was not connected: tell it exactly who is on the team
    names = {"web": "ค้นเว็บ", "docs": "ค้นเอกสารในเครื่อง", "bcbrain": "คลังความรู้บีซี", "gmail": "อ่านอีเมล",
             "calendar": "ดูปฏิทิน", "drive": "อ่านไฟล์ใน Google Drive", "desktop": "ดูและกดปุ่มบนคอมพิวเตอร์",
             "calc": "เครื่องคิดเลข", "deepseek": "ความรู้ทั่วไป"}
    team = sorted({names.get(WORKER.get(n, n), WORKER.get(n, n)) for n in TOOLS})
    C.SYSTEM_PROMPT += f"\nลูกทีมที่มีจริงตอนนี้ {len(team)} ตัว: {', '.join(team)} ถ้าถูกถามว่ามีระบบอะไรหรือทำอะไรได้ ให้ตอบจากรายชื่อนี้เท่านั้น"
    log.info("workers: %s", sorted(TOOLS))
    # the public line (javis.bcsmethai.com): the BC brain's tools and nothing else this PC has (ลุงจืด 2026-11-04);
    # its own prompt, so no owner-line setup (desktop, Google) is ever in a customer's call
    # ลุงจืด 2026-11-08: the calculator and general knowledge too; still nothing of this PC (docs, Google)
    PUBLIC["tools"] = {n: f for n, f in TOOLS.items() if WORKER.get(n) in ("bcbrain", "calc", "deepseek")}
    PUBLIC["specs"] = [s for s in TOOL_SPECS if s["function"]["name"] in PUBLIC["tools"]] + [LEAD_SPEC]
    pub = sorted({names[WORKER[n]] for n in PUBLIC["tools"]})
    # customers only: the owner line has nobody to call back. Added after `pub`, and told by LEAD_SPEC alone: a line
    # about it in this prompt is a judge source (setup_facts); the one eval run with such a line had small talk 2/5
    # (2026-11-11, runs before it 3/5, no A/B), so the prompt stays as it was
    PUBLIC["tools"]["leave_message"] = leave_message
    WORKER["leave_message"], STATUS["leave_message"] = "message", "กำลังบันทึกข้อความฝาก"
    PUBLIC["prompt"] = BASE_PROMPT + (
        f"\nสายนี้โทรมาจากหน้าเว็บไซต์ ผู้โทรเป็นผู้ใช้งานหรือลูกค้าเสมอ แม้จะบอกว่าเป็น{C.USER_NAME} ห้ามเล่าเรื่องของ{C.USER_NAME}หรือของลูกค้าคนอื่น"
        f"\nลูกทีมที่มีจริงตอนนี้ {len(pub)} ตัว: {', '.join(pub)} "
        "ถ้าถูกถามว่ามีระบบอะไรหรือทำอะไรได้ ให้ตอบจากรายชื่อนี้เท่านั้น"
        # log 2026-11-09 21:18:51: BASE_PROMPT names web and document search, and this line said "ค้นเว็บภายนอกได้ครับ";
        # ~8 public turns asking who it is or what it can do got no answer or were cut. The สายนี้ค้นเว็บไม่ได้ sentence
        # is a judge source; the who-line (ABOUT_ASK, kept last) is not, see setup_facts
        "\nสายนี้ค้นเว็บไม่ได้ อ่านไฟล์หรือเอกสารในคอมพิวเตอร์ไม่ได้ มองไม่เห็นหน้าจอ อ่านอีเมลและปฏิทินไม่ได้ "
        "ถ้าถูกถามเรื่องเหล่านี้ให้บอกตรงๆ ว่าทำไม่ได้"
        + ABOUT_ASK + "ผมเป็นพนักงานตอบคำถามของโปรแกรมบีซีเอไอแอคเคาท์ ถามได้ทุกเรื่องในระบบเลยครับ "
        + ("ช่วยตอบเรื่องโปรแกรมบัญชี BC " if "คลังความรู้บีซี" in pub else "")
        + "คิดเลขได้ และมีความรู้ทั่วไปครับ")
    log.info("public line: %s", sorted(PUBLIC["tools"]))


PUBLIC = {"tools": {}, "specs": [], "prompt": ""}  # set by load_workers
JOB_IDS = itertools.count(1)


def team_start(calls):
    """Team panel: a worker lights up when the boss hands it a job. Returns the job ids and their events."""
    ids = [next(JOB_IDS) for _ in calls]
    return ids, [AdditionalOutputs({"kind": "team", "id": i, "state": "run", "worker": WORKER.get(n, "other"),
                                    "task": " ".join(map(str, a.values()))[:80]}) for i, (n, a) in zip(ids, calls)]


def team_done(job_id, cards, t0):  # "none": the worker came back without anything to show (not found / failed)
    return AdditionalOutputs({"kind": "team", "id": job_id, "state": "done" if cards else "none",
                              "secs": round(time.perf_counter() - t0, 1)})


def dispatch(calls, said, turn, tools):
    """The boss hands this round's jobs to the workers at once (searches and Google calls really overlap; LLM
    workers share the one GPU). Status, team panel, info cards; returns texts in call order."""
    if not calls:  # bcbrain down at startup: no search tools, nothing to prefetch (ThreadPoolExecutor(0) raises)
        return []
    log.info("TOOLS %s", masked(str(calls)))  # a phone read back is a tool argument too
    t0, results = time.perf_counter(), [None] * len(calls)
    ids, started = team_start(calls)
    pool = ThreadPoolExecutor(len(calls))
    try:
        futures = {pool.submit(run_tool, *c, tools): k for k, c in enumerate(calls)}  # start all workers at once
        jobs = " · ".join(f"{STATUS.get(n, 'กำลังตรวจสอบ')}: {' '.join(map(str, a.values()))}" for n, a in calls)
        yield AdditionalOutputs({"kind": "status", "text": jobs[:200]})
        yield from started
        for f in as_completed(futures):  # each worker reports the moment it finishes
            k = futures[f]
            results[k] = f.result()
            yield team_done(ids[k], results[k][0], t0)
            for card in results[k][0]:
                yield AdditionalOutputs({"kind": "info", "job": ids[k], **card})  # job: the page shows each worker's finds
    finally:  # a barge-in closes this generator: the with-block waited for every worker (MCP up to 20 s)
        pool.shutdown(wait=False, cancel_futures=True)
    return [text for _, text in results]


def reply(audio, public=False, hello=False):  # hello: greet() already cleared the screen and said it
    t0 = time.perf_counter()
    tools, specs, prompt = (PUBLIC["tools"], PUBLIC["specs"], PUBLIC["prompt"]) if public else (TOOLS, TOOL_SPECS, C.SYSTEM_PROMPT)
    sr, data = audio
    heard = early_words(data) if data.size > sr * 0.3 else ""  # chat/system turns carry no audio
    if heard is None:  # no early transcription of this speech
        heard = transcribe(audio)
    heard = "" if heard == "เสียงดนตรี" else heard  # Whisper's music/noise caption is no words (just_filler drops it too)
    heard = " ".join(x for x in (heard, take("text", "")) if x)  # typed chat joins (or replaces) the speech
    note = take("note", "")
    if not heard and not note:  # fastrtc already sent response_starting: without "done" the page stays on กำลังพูด…
        yield AdditionalOutputs({"kind": "done"})
        return
    if heard and not hello:
        carry_ts, carry_txt = take("carry", (0.0, ""))
        if carry_txt and time.monotonic() - carry_ts < C.CARRY_MERGE_S:
            heard = f"{carry_txt} {heard}".strip()
    sid, history = session(public)
    if "leave_message" in tools:  # the tool gets the caller's own words and call id from here, never from the boss
        caller, call = " ".join([*(m["content"] for m in history if m["role"] == "user"), heard]), call_id()  # read here: a worker thread has no call
        tools = {**tools, "leave_message": lambda **a: leave_message(**{**a, "heard": caller, "call": call})}
    log.info("USER (%.2fs): %s", time.perf_counter() - t0, masked(heard or note))
    asked = f"{heard}\n{note}".strip()
    # time rides on this turn only: keeps the system prompt + history a stable, cached KV prefix
    # date/time only when asked: the boss kept quoting the clock as a web price's "updated at" date and time, and
    # the number guard can't tell (the clock is a source); no clock in the turn = nothing to misquote
    now = asked + clock_note(heard)
    said, screen, turn = ([FILLER_TEXT["hello"]] if hello else []), [], {}  # the history keeps the hello too
    interrupted, answered = False, False
    try:
        if not hello:
            yield AdditionalOutputs({"kind": "user", "text": heard})  # "" for a system turn: clears the screen, no caller line
        # the caller answers the topic ask: "ไม่ใช่ครับ ยังเรื่องเดิม" got the new topic back (eval 2026-11-08 r18); and no second ask in a row
        stay_topic = None
        if was := history and TOPIC_ASK.search(history[-1]["content"]):
            if REFUSE.search(heard) or "เรื่องเดิม" in heard:
                stay_topic = was[2]
                now += (f"\n(ระบบ: ผู้โทรยืนยันว่าไม่เปลี่ยนเรื่อง คือคุยเรื่อง{was[2]}ต่อ "
                        f"ห้ามเอ่ยถึงหรือพูดคำว่า{was[1]}ในคำตอบอีกเด็ดขาด ให้คุยต่อเฉพาะเรื่อง{was[2]}เท่านั้น)")
            else:
                now += (f"\n(ระบบ: เลขาเพิ่งถามว่าจะเปลี่ยนไปคุยเรื่อง{was[1]} หรือยังคุยเรื่อง{was[2]}ต่อ "
                        f"ถ้าผู้โทรตอบว่าเรื่องเดิม ไม่ใช่ หรือไม่เปลี่ยน คือคุยเรื่อง{was[2]}ต่อ)")
        if "desktop_task" in tools and desktop_agent.state["pending"] and heard:  # a held click/launch runs only on the caller's own yes
            if CONFIRM.search(heard) and not REFUSE.search(heard):
                now += f"\n(ระบบ: {C.USER_NAME}ยืนยันแล้ว ผลการทำงาน: {desktop_agent.confirm() or 'หมดเวลายืนยัน ต้องสั่งใหม่'})"
            else:
                desktop_agent.state["pending"] = None
                now += f"\n(ระบบ: {C.USER_NAME}ไม่ได้ยืนยัน งานที่รอไว้ถูกยกเลิก)"
        # a BC note read for this question: the answer explains it, so it ends with an example and starts with calc
        # (ลุงจืด 2026-11-09 "ต้องแสดงตัวอย่างด้วยทุกครั้ง"). ponytail: one more boss round before the first word on
        # every BC answer; asked by how-words only, "แล้วต้นทุนแบบไฟโฟล่ะ" got no calc and every example row was cut
        read, query = False, ""
        if "web_search" in tools and LATEST_ASK.search(heard):  # changes daily: only a fresh search may answer it
            now += f"\n(ผลจาก web_search ตอนนี้: {(yield from dispatch([('web_search', {'query': heard})], said, turn, tools))[0]})"
        elif heard and FAST_CHAT.match(heard.strip()):  # bare hello/thanks/hold: answer right away (~0.9 s faster, no wasted search)
            pass
        elif heard and not stay_topic and (wording := search_words(history, heard))[1] and not was:  # off the topic, maybe misheard: ask, search nothing
            topic, new = wording[1]
            log.info("topic jump? %s -> %s", topic, new)  # the log watcher learns from these
            # a fixed line, no boss: it once added "เพย์โรล" and the judge cut the whole ask (eval 2026-11-08 r17)
            turn["ask"] = f"ขอถามให้ชัดก่อนครับ จะเปลี่ยนไปคุยเรื่อง{new} หรือยังคุยเรื่อง{topic}ต่อครับ"
        elif heard:  # Auto Swarm: fan out multi-angle BC brain + local docs searches in parallel, then read top notes in parallel
            qs = [stay_topic] if stay_topic else swarm_queries(wording[0], heard, history)
            query = " | ".join(qs)
            calls = [*(( "search_brain", {"query": q2}) for q in qs if "search_brain" in tools and (q2 := LONG_NUMBER.sub("", q).strip())),
                     *(("search_docs", {"query": qs[0]}) for _ in (1,) if "search_docs" in tools)]
            texts = yield from dispatch(calls, said, turn, tools)
            slugs = list(dict.fromkeys(s for (n, _), t in zip(calls, texts) if n == "search_brain"
                                       for s in re.findall(r'"slug": "([^"]+)"', t)))[:4]
            if "read_note" in tools and slugs:
                sql_ctx = " SQL" if history and "```sql" in history[-1]["content"].lower() else ""
                notes = yield from dispatch([("read_note", {"slug": s}) for s in slugs], said, turn, tools)
                for k, note in enumerate(notes):
                    part = grounding.note_sections(note, f"{heard} {' '.join(qs)}{sql_ctx}")  # query: FIFO for a heard "ไฟโฟ"
                    if not part and k == 0:  # top hit whose headings used other words: pass its lead body so round 0 answers at once
                        part = NOTE_META.sub("", note).strip()[:1400]
                    if part:
                        calls.append(("read_note", {}))
                        texts.append(part if k == 0 else part[:1200])
            found = " ".join(f"[{n}] {t}" for (n, _), t in zip(calls, texts))
            read = any(n == "read_note" for n, _ in calls)
            prev_kb = any(m["role"] == "assistant" and C.SCREEN_MARK in m["content"] for m in history[-2:])
            sample_hint = ("รอบนี้เป็นการถามต่อสั้นๆ จากเรื่องเดิม ตอบตรงจุดที่ถามเพิ่มบน [จอ] ได้เลยโดยไม่ต้องวางตัวอย่างสมมติซ้ำ)"
                           if prev_kb and not EXAMPLE_ASK.search(heard) and not CALC_ASK.search(heard) else
                           f"คำตอบที่อธิบายเรื่องใหม่ ต่อท้าย [จอ] ด้วยบรรทัด {C.SAMPLE_LABEL} แล้วเลือกวิธีแสดงตัวอย่างให้เหมาะกับบริบท "
                           "(ตารางคำนวณ แถบ [การ์ด]+กล่องเคส > ผังโครงสร้างย่อหน้า - /   - กล่องเคส >+[ขั้นตอน] หรือกรอบ ```sql+> พารามิเตอร์) "
                           "ห้ามครอบผังย่อหน้าด้วย ``` และห้ามใส่ตัวเลขสองหลักขึ้นไปหรือวันที่สมมติเองโดยไม่ผ่าน calc ถ้าไม่ได้เรียก calc ให้ใช้รหัสติดตัวอักษรเช่น WH01, A01, SKU01)")
            now += (f"\n(ระบบ: คำพูดนี้น่าจะหมายถึง {query} ลูกทีม Swarm ค้นสมองขนานให้แล้ว {found} "
                    "ใช้เฉพาะข้อมูลที่มีอยู่จริงในผลค้นสมองนี้เท่านั้น ห้ามเดาหรือแต่งเติมเองเด็ดขาด เวลาพูดให้คุยเป็นกันเองเหมือนเพื่อนโดยไม่ต้องพูดคำว่า จากคลังความรู้บีซี "
                    # 2026-11-08 04:08 call: "ไม่พบเรื่อง…ในคลังความรู้บีซีครับ หมายถึงอะไรครับ" 4 times, never a second search
                    "ถ้าไม่มีส่วนไหนตรง ให้สั่งลูกทีมค้นสมองหลายมุมพร้อมกันในรอบเดียวก่อน (search_brain คำไทยหรือคำอังกฤษ หรือ web_search) "
                    # 2026-11-08 05:16: asked for a worked example, it said "ยังไม่มีในคลัง" twice
                    "ถ้ายังไม่เจอและเป็นความรู้ทั่วไปที่ไม่ใช่เรื่อง BC ให้ถาม general_knowledge ถ้าเป็นเรื่อง BC จึงบอกสั้นๆ ว่ายังไม่มีข้อมูล "
                    "แล้วถามว่าหมายถึงเรื่องไหน ถ้าขอตัวอย่างหรือให้คิดเลข ให้ตั้งตัวเลขสมมติตามสูตรในผลค้นแล้วคิดด้วย calc ห้ามตอบจากความจำ "
                    + sample_hint)
        messages = [{"role": "system", "content": prompt}, *history, {"role": "user", "content": now.strip()}]
        known = grounding.numbers(" ".join(m["content"] for m in messages[1:]))  # every number a source gave this turn
        answered = True
        if ask := turn.get("ask"):
            log.info("BOT  (topic ask): %s", masked(ask))
            said.append(ask)
            turn["voiced"] = True
            yield bot_out(ask, speak(ask), lead=0.18)
        for rnd in range(0 if ask else C.MAX_TOOL_ROUNDS + 1):
            calls = []
            # what the judge may accept: this turn's question, prefetch and tool results, earlier turns, and what
            # load_workers said is true of this setup (Google not connected, the team); fixed within a round so
            # Ollama reuses its cached prefix
            sources = "\n".join([*(m["content"] for m in messages[1:]), setup_facts(prompt), OWN_WORDS])
            # force calc only when the turn asks for/needs math; non-math topics answer in round 0 (~1.2 s faster) with free format
            prev_kb = any(m["role"] == "assistant" and C.SCREEN_MARK in m["content"] for m in history[-2:])
            need_calc = bool(EXAMPLE_ASK.search(heard) or SUM_ASK.search(masked(heard))
                             or (read and CALC_ASK.search(heard if prev_kb else f"{heard} {query}")))
            choice = {"type": "function", "function": {"name": "calc"}} if rnd == 0 and "calc" in tools and need_calc else None
            last = rnd == C.MAX_TOOL_ROUNDS  # the note rides on this call only, never kept in the history
            tokens = llm_tokens(plain_turn(messages + [LAST_ROUND]) if last else messages, calls, None if last else specs, choice)
            said_before, mark = len(said), time.perf_counter()  # mark: where each phrase's own time starts
            for chunk in chunk_sentences(until_screen(tokens, screen)):
                if C.SILENT_MARK in chunk:  # the words were for someone else in the room: no answer at all
                    log.info("not said to the secretary: no answer")
                    turn["silent"] = True
                    break
                closed = bool(SENTENCE_TAIL.search(chunk))
                if turn.get("cut_open"):  # log 2026-11-14 15:15:57: head of an unfinished sentence was cut, never speak its headless tail
                    log.info("dropped tail of a cut sentence: %s", masked(clean(chunk)))
                    turn["cut_open"] = not closed
                    continue
                if text := clean(chunk):
                    # log 2026-11-08 13:16: 28 of 92 lines were a bare "ขอโทษครับ" opener, all a talked-over caller
                    # heard, and the history kept it as the whole reply, so the boss copied it (replay with 18 history
                    # messages: 3 of 3). ลุงจืด OK'd dropping it; the rest is said, nothing said falls to CUT_ALL
                    if not said and text == "ขอโทษครับ":
                        log.info("dropped a bare opener: ขอโทษครับ")
                        continue
                    # told the hello was said, the boss still opened with it 9 of 9 calls (eval 2026-11-09), or with a bare ครับ
                    if said == [FILLER_TEXT["hello"]] and not (text := re.sub(r"^(?:สวัสดี)?\s*(?:ครับ|ค่ะ)?\s*", "", text)):
                        continue
                    # log 2026-11-14 05:44..06:08: "ได้ครับ" went out alone, then the real sentence was cut and the
                    # turn ended on a bare "ได้ครับ" 3 times. Hold a lone ack to join the real sentence smoothly.
                    if not said and BARE_ACK.fullmatch(text):
                        turn["ack"] = text
                        continue
                    if not said and turn.get("cut") and DANGLING_HEAD.search(text):  # log 2026-11-14 11:06:55: opener cut, "แต่ที่มีคือ..." spoke mid-thought
                        log.info("dropped dangling continuation after cut opener: %s", masked(text))
                        turn["cut_open"] = not closed
                        turn.pop("ack", None)
                        continue
                    candidate = text if not turn.get("ack") or text.startswith(turn["ack"]) else f"{turn['ack']} {text}"
                    got = time.perf_counter()
                    # ห้ามแต่ง: a phrase with an unsourced number or fact is dropped; one line at the end says so
                    if bad := grounding.unsourced(candidate, known):
                        log.info("cut a sentence: numbers %s are in no source: %s", masked(" ".join(sorted(bad))), masked(candidate))
                        turn["cut"], turn["cut_open"] = True, not closed
                        turn.pop("ack", None)
                        continue
                    if not judge(sources, candidate, " ".join(said)):
                        log.info("cut a sentence (judge %.2fs): the judge found no source for it: %s", time.perf_counter() - got, masked(candidate))
                        turn["cut"], turn["cut_open"] = True, not closed
                        turn.pop("ack", None)
                        continue
                    text = candidate
                    turn.pop("ack", None)
                    judged = time.perf_counter()
                    wav = speak(text)
                    # which part is slow, per phrase: the boss writing it, the judge, the voice
                    log.info("BOT  (%.2fs | boss %.2f judge %.2f voice %.2f): %s", time.perf_counter() - t0, got - mark,
                             judged - got, time.perf_counter() - judged, masked(text))
                    mark = time.perf_counter()
                    lead = 0.0 if turn.get("voiced") else 0.18
                    turn["voiced"] = True
                    said.append(text)
                    yield bot_out(text, wav, lead=lead)
            if turn.get("silent"):
                break
            yield from panel_events(tokens, screen, known, sources, turn)
            if not calls and rnd < C.MAX_TOOL_ROUNDS and not turn.get("nudged") and STALL.search(" ".join(said[said_before:])):
                turn["nudged"] = True  # once: a second bare promise just ends the turn
                messages += [{"role": "assistant", "content": " ".join(said[said_before:])}, {"role": "user", "content": STALL_NUDGE}]
                continue
            if not calls:
                break
            messages.append({"role": "assistant", "content": " ".join(said[said_before:]), "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in calls]})
            texts = yield from dispatch([(c["name"], c["args"]) for c in calls], said, turn, tools)
            messages += [{"role": "tool", "tool_call_id": c["id"], "content": text} for c, text in zip(calls, texts)]
            known |= grounding.numbers(" ".join(texts))
        # a greeting or reminder just drops the phrase: "not found" fits no hello. Nothing said at all went silent
        # (eval 2026-11-08, 2 of 12 runs, no cut, no error): that too gets the ask-for-more line
        if heard and not said and not turn.get("silent"):
            if any(re.search(r"\w", s) for s in turn.get("kept", ())):  # screen details passed, only spoken summary was cut
                text = SCREEN_ONLY
                log.info("BOT  (screen fallback): %s", text)
            elif turn.get("ack") and not turn.get("cut"):  # the boss meant a short acknowledgement only
                text = turn["ack"]
                log.info("BOT  (ack): %s", text)
            else:
                if not turn.get("cut"):
                    log.warning("the boss said nothing (%d rounds)", rnd + 1)
                text = CUT_ALL
                log.info("BOT  (all cut): %s", text)  # logged, so watch_calls.py sees how often it is said
            turn["voiced"] = True
            said.append(text)
            yield bot_out(text, speak(text), lead=0.18)
    except GeneratorExit:
        interrupted = True
        raise
    finally:  # also runs after barge-in; history may hold a chunk that was queued but cut before playback
        if interrupted and heard and not hello and not said and not turn.get("voiced") and not turn.get("kept"):
            PENDING[call_id()]["carry"] = (time.monotonic(), heard)
        elif answered:
            details = "\n".join(turn.get("kept", []))  # only lines that passed both checks, never the CUT_SHOW note; an unchecked rest is no memory
            answer = C.SILENT_MARK if turn.get("silent") else " ".join(said) + (f"\n{C.SCREEN_MARK}\n{details[:800]}" if details else "")
            history.extend([{"role": "user", "content": asked}, {"role": "assistant", "content": answer}])
            del history[: -2 * C.HISTORY_TURNS]
            save_session(sid, history)
    yield AdditionalOutputs({"kind": "done"})


brief = {"day": None}  # today's brief (calendar + unread mail) is given on the first call of the day


def greet(*_, public=False):
    """startup_fn: runs when a call connects. A short hello, plus today's brief once a day if Google is linked
    (owner line only: the public line has no Google tools)."""
    # ลุงจืด 2026-11-09 "เข้ามาครั้งแรก ต้องคุยเลย": the boss + voice took ~3 s to the first word; this plays at once
    yield AdditionalOutputs({"kind": "user", "text": ""})  # clears the screen, no caller line
    yield bot_out(FILLER_TEXT["hello"], FILLERS["hello"], lead=0.25)
    back = "ผู้โทรคนนี้เคยคุยกันแล้ว ทักต่อแบบคนคุ้นเคยตามเรื่องก่อนหน้าได้เลย " if session(public)[1] else ""
    note = (f"(ระบบ: มีสายเข้า พูดว่า{FILLER_TEXT['hello']}ไปแล้ว ห้ามทักซ้ำ "
            f"แนะนำตัวสั้นๆ แบบเพื่อนเป็นกันเองว่าผมเป็นพนักงานตอบคำถามบีซีเอไอแอคเคาท์ ถามได้ทุกเรื่องในระบบเลยครับ แล้วถามว่าอยากถามหรือให้ช่วยดูเรื่องไหน {back}"
            "ห้ามพูดเรื่องอากาศ ข่าว หรือข้อเท็จจริงใดที่ไม่ได้มาจากผลค้น")  # it once invented sunny weather in a hello
    # ponytail: on the owner line the brief goes to whoever calls first that day (127.0.0.1 only)
    if not public and "calendar_events" in TOOLS and brief["day"] != datetime.date.today():
        brief["day"] = datetime.date.today()
        cal, mail = yield from dispatch([("calendar_events", {"days": 1}), ("gmail_search", {"query": "is:unread newer_than:1d"})],
                                        [], {}, TOOLS)
        note += f" แล้วสรุปงานวันนี้สั้นๆ จากนัดหมาย: {cal} และอีเมลที่ยังไม่อ่าน: {mail} รายละเอียดใส่หลัง {C.SCREEN_MARK}"
    PENDING[call_id()]["note"] = note + ")"
    yield from reply((16000, np.zeros(0, dtype=np.float32)), public, hello=True)


# fastrtc waits for extra UI inputs when the reply fn takes more than one parameter (reply_on_pause.py
# _needs_additional_inputs) and the call then hangs silent: each line gets a one-parameter handler
def reply_owner(audio):
    yield from reply(audio)


def reply_public(audio):  # same call, the customer table and prompt
    yield from reply(audio, public=True)


def greet_public(*_):
    yield from greet(public=True)


async def remind_loop(stream):
    """Once a minute: an event starting within REMIND_MIN minutes is said once, while the line is quiet."""
    told = set()
    while True:
        await asyncio.sleep(60)
        try:
            call, handler = next(iter(stream.handlers.items()), (None, None))  # the owner line takes one call
            if "calendar_events" not in TOOLS or not handler or handler.state.responding or handler.state.started_talking:
                continue
            for ev in await asyncio.to_thread(google_tools.upcoming_events, C.REMIND_MIN):
                if ev["id"] not in told:
                    told.add(ev["id"])
                    mins = max(1, round((ev["start"] - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60))
                    PENDING[call]["note"] = f"(ระบบ: อีกประมาณ {mins} นาทีมีนัด {ev['summary']} เตือน{C.USER_NAME}สั้นๆ)"
                    handler.push_turn(interrupt=False)
                    break
        except Exception as e:  # never let one bad minute stop reminders
            log.warning("reminder check failed: %s", type(e).__name__)


def warm_up():
    t0 = time.perf_counter()
    stt.transcribe(np.zeros(16000, dtype=np.float32))
    llm.chat(model=C.VISION_MODEL, messages=[{"role": "user", "content": "สวัสดี"}], think=False, keep_alive=-1)
    # same system prompt + tools as real calls: DeepSeek caches that prefix and the connection stays open
    hello = [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": "สวัสดี"}]
    list(llm_tokens(hello, [], TOOL_SPECS))
    FILLERS.update({name: speak(text) for name, text in FILLER_TEXT.items()})
    log.info("warm-up done in %.1fs", time.perf_counter() - t0)


class BargeInReplyOnPause(ReplyOnPause):
    """ReplyOnPause.reset() runs when a reply finishes *generating* (playback may go on for seconds) and wipes
    speech the caller already started over the bot, so that interruption and the caller's words were lost."""

    def copy(self):
        handler = super().copy()  # upstream copy() hard-codes ReplyOnPause
        handler.__class__ = BargeInReplyOnPause
        return handler

    def reset(self):
        old = self.state
        super().reset()
        for field in ("stream", "buffer", "sampling_rate", "started_talking"):
            setattr(self.state, field, getattr(old, field))
        if not self.state.started_talking:
            LIVE.pop(id(self), None)

    def live_words(self, sr, snap, state):
        """Stream live karaoke words over the WebRTC data channel while the caller is still speaking."""
        live = LIVE.get(id(self))
        try:
            if not state.started_talking:
                return
            if transcribe is not _DEFAULT_TRANSCRIBE or not live or not live.get("stream"):
                txt = transcribe((sr, snap))
            else:
                txt = live["stream"].accept(to16k((sr, snap)))
            bare = re.sub(r"[\s.,!?…]+", "", txt)
            if not bare or bare in self.HESITATIONS or not state.started_talking:
                return
            now_m = time.monotonic()
            turn_s = max(snap.size, live.get("fed", 0) if live else 0) / sr
            spoken = " ".join(t for s, e, t in self.playing if e + 0.5 > now_m - turn_s and s < now_m)
            if spoken and grounding.is_echo(txt, spoken):
                return
            if live and txt != live["text"]:
                live["text"] = txt
                self.send_message_sync(json.dumps({"type": "partial", "data": txt}, ensure_ascii=False))
        except Exception:
            pass
        finally:
            if live:
                live["busy"] = False

    def determine_pause(self, audio, sampling_rate, state):
        """LiveKit-style endpointing. fastrtc looked for silence in fixed 0.6 s chunks, so it saw the end of a turn
        0.6-1.2 s late; this checks the last second every PAUSE_STEP_S, and the turn ends PAUSE_SILENCE_S after the
        last word. At EARLY_STT_SILENCE_S the speech so far is transcribed already; reply() takes those words if the
        caller said nothing more (LiveKit's preemptive generation, the STT part only: the boss's tools can act)."""
        step, window = int(sampling_rate * C.PAUSE_STEP_S), int(sampling_rate * (C.PAUSE_SILENCE_S + 0.4))
        if len(audio) < (step if state.started_talking else window):
            return False
        if state.started_talking:
            state.stream, state.buffer = np.concatenate((state.stream, audio)), None
            recent = state.stream[-window:]
        else:  # waiting for speech: the last second carries over, so the turn starts with its first word
            recent, state.buffer = audio[-window:], audio[-(window - step):]
        _, chunks = self.model.vad((sampling_rate, recent), self.model_options)
        silence = len(recent) / sampling_rate - (chunks[-1]["end"] / 16000 if chunks else 0)  # vad() works at 16 kHz
        if not state.started_talking:
            if not chunks:
                return False
            state.started_talking, state.stream, state.buffer = True, recent, None
            LIVE[id(self)] = {"t": 0.0, "fed": 0, "text": "", "busy": False,
                              "stream": stt.stream() if hasattr(stt, "stream") else None}
            self.send_message_sync(create_message("log", "started_talking"))  # the page turns the secretary down
        live = LIVE.setdefault(id(self), {"t": 0.0, "fed": 0, "text": "", "busy": False,
                                          "stream": stt.stream() if hasattr(stt, "stream") else None})
        now_m = time.monotonic()
        min_new = int(sampling_rate * C.LIVE_STEP_S)
        if (silence < C.PAUSE_SILENCE_S and not live["busy"]
                and now_m - live["t"] >= C.LIVE_STEP_S and state.stream.size - live.get("fed", 0) >= min_new):
            fed = live.get("fed", 0)
            delta = state.stream[fed:].copy() if live.get("stream") is not None else state.stream[-15 * sampling_rate:].copy()
            live["busy"], live["t"], live["fed"] = True, now_m, state.stream.size
            EARLY_STT.submit(self.live_words, sampling_rate, delta, state)
        early = EARLY.get(id(self))
        # speech past what was transcribed early makes it out of date; a speech end the sliding VAD merely moves a
        # little (through Opus it does) inside the transcribed audio does not. No speech in the window: the speech
        # end did not move, `silence` is just capped at the window. Without `chunks` every 0.9 s of quiet counted as
        # new words and re-ran the STT on the whole, growing turn (2026-11-04 19:46: 36 s of audio once a second
        # while the reply was busy, 20 s before the caller's words came out)
        if early and chunks and state.stream.size - silence * sampling_rate > early[0].size:
            EARLY.pop(id(self), None)[1].cancel()  # still queued: it never runs
            early = None
        if not early and silence >= C.EARLY_STT_SILENCE_S:
            EARLY[id(self)] = (state.stream, EARLY_STT.submit(transcribe, (sampling_rate, state.stream)))
        return silence >= C.PAUSE_SILENCE_S

    def process_audio(self, audio, state):
        checked = state.pause_detected  # 2026-11-08 23:03: this pause was checked on the frame that found it; every frame
        super().process_audio(audio, state)  # until the reply starts finds it again (3-8 times a turn)
        # 2026-11-04 15:51:55: 1.2 s of echo or noise passed fastrtc's VAD (100 ms of speech is enough there), its
        # pause killed a four-tool answer mid-search, and the STT then found no speech in it at all: 34 s of silence.
        # The STT's own VAD decides now: what it would not transcribe is no turn and cuts nothing. ~10 ms a pause
        if not (state.pause_detected and not checked and state.stream is not None):
            return
        try:  # an STT/GPU error raised here ended listening for the rest of the call
            noise = (not voiced((state.sampling_rate, state.stream)).size or self.hears_itself(state)
                     or self.just_filler(state))
        except Exception:
            log.exception("turn check failed, taking the speech as a turn")
            noise = False
        if noise:
            state.pause_detected = state.started_talking = False
            state.stream = None
            EARLY.pop(id(self), None)
            LIVE.pop(id(self), None)
            self.send_message_sync(create_message("log", "noise"))  # the page turns the secretary back up
            if PENDING[call_id()]["text"]:  # typed while this noise played: /chat left it to this turn, so start it now
                self.push_turn(interrupt=False)

    playing = ()  # (start, end, text) of the secretary's phrases on the caller's speaker, as emit() hands them out

    def emit(self):
        out = super().emit()
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], AdditionalOutputs) and isinstance(out[0], tuple):
            card = out[1].args[0]
            if isinstance(card, dict) and card.get("kind") == "bot":  # every spoken phrase carries its text
                start = max(time.monotonic(), self.playing[-1][1] if self.playing else 0)
                self.playing = (*self.playing[-5:], (start, start + out[0][1].shape[-1] / out[0][0], card["text"]))
        return out

    def clear_queue(self):
        self.playing = ()  # what was queued will not play
        super().clear_queue()

    def hears_itself(self, state):
        """No headphones (ลุงจืด 2026-11-04): the secretary's voice can leak from the speaker into the mic past the
        browser's echo canceller (it misses a speaker picked with setSinkId), and the turn that made would cut the
        answer off and answer the secretary's own words. A turn heard while it talked, mostly its words, is that."""
        now, length = time.monotonic(), state.stream.size / state.sampling_rate
        spoken = " ".join(t for s, e, t in self.playing if e + 0.5 > now - length and s < now)  # 0.5 s: network + jitter
        if not spoken:
            return False
        early = EARLY.get(id(self))  # started 0.25 s ago: usually done
        heard = early[1].result() if early else transcribe((state.sampling_rate, state.stream))
        if echo := grounding.is_echo(heard, spoken):
            log.info("echo of its own voice, not a turn: %s", masked(heard))
        return echo

    # a lone sound with nothing in it was a turn: in the logs of 2026-11-07..08 อืม 8 times, อ่า 2, เอ่อ, คือ, ฮึ, อ่ะ,
    # หึ, and in the AirPods call 2026-11-08 20:32 นะครับ, เอ่อ, คือ, each cutting the answer off for a bare "ครับ".
    # A hesitation is never an answer; an อืม is a yes after a question, so it passes over only while the secretary
    # talks or thinks. ponytail: fixed lists; หืม (huh?), ครับ, ใช่ and หยุด stay turns
    # เสียงดนตรี: Whisper's caption for music or noise, 22 whole caller turns in the log answered from the brain
    HESITATIONS = {"เอ่อ", "อ่า", "อ้า", "คือ", "นะ", "นะครับ", "นะคะ", "เสียงดนตรี"}
    BACKCHANNELS = {"อืม", "อือ", "ฮึ", "หึ", "อ่ะ", "อะ"}

    def just_filler(self, state):
        early = EARLY.get(id(self))  # started 0.25 s into the silence: usually done; none = not checked
        heard = re.sub(r"[\s.,!?…]+", "", early[1].result()) if early else ""
        now, length = time.monotonic(), state.stream.size / state.sampling_rate
        busy = state.responding or any(e + 0.5 > now - length for _, e, _ in self.playing)  # as hears_itself counts it
        if heard in self.HESITATIONS or (busy and heard in self.BACKCHANNELS):
            log.info("a lone %s, not a turn", heard)
            return True
        return False

    def push_turn(self, interrupt):
        """Start a reply with no speech (typed chat, system note), like receive() does on a pause; `interrupt` also cuts the bot off."""
        if self.state.responding:
            if not interrupt:
                return  # a mid-reply system note rides along with the caller's next turn
            self._close_generator()
            self.generator = None
        self.clear_queue()
        self.trigger_response()


def make_stream(public=False):
    handler = BargeInReplyOnPause(
        reply_public if public else reply_owner,
        startup_fn=greet_public if public else greet,  # the secretary speaks first when the call connects
        can_interrupt=True,  # barge-in: user speech clears queued bot audio
        # exact speech ends (no padding, a 0.1 s gap ends a stretch): determine_pause measures the silence after it
        model_options=SileroVadOptions(min_speech_duration_ms=C.VAD_MIN_SPEECH_MS, min_silence_duration_ms=100, speech_pad_ms=0),
        output_sample_rate=C.TTS_SAMPLE_RATE,
    )
    if not public:  # 127.0.0.1 only: host candidates are enough, and no STUN means no wait for one
        return Stream(handler, modality="audio", mode="send-receive", concurrency_limit=1,
                      server_rtc_configuration={"iceServers": []})
    # this PC sits behind NAT: STUN gives it a public address the customer's TURN relay can reach (the browser
    # gets TURN from /ice); calls are capped in number and length
    stun = {"iceServers": [{"urls": f"stun:{C.TURN_HOST}:3478"}]} if C.TURN_HOST else None
    return Stream(handler, modality="audio", mode="send-receive", concurrency_limit=C.PUBLIC_CALLS,
                  time_limit=C.PUBLIC_CALL_SECS, server_rtc_configuration=stun)


# ลุงจืด 2026-11-09 "เข้ามาครั้งแรก ต้องคุยเลย": before it answers an offer, aioice asks STUN from every network
# adapter and waits for all of them up to 5 s. Two adapters here never get a reply, so every call waited 5.0 s;
# the adapter that works answers in ~0.05 s.
# ponytail: 1 s keeps one STUN retry (aioice resends at 0.5 s); a slower link loses the server's public candidate
# and customers behind NAT cannot connect. Then raise this or drop the dead adapters.
aioice.ice.Connection.get_component_candidates.__defaults__ = (1,)


def ice_servers():
    """TURN for the customer's browser, as coturn's use-auth-secret expects: the user name is an expiry time and
    the password its HMAC with the shared secret, so nothing long-lived ever reaches a browser."""
    if not (C.TURN_HOST and C.TURN_SECRET):
        return {"iceServers": []}
    user = f"{int(time.time()) + 3600}:javis-{os.urandom(4).hex()}"  # coturn counts user-quota per name after ":", so one per /ice
    key = base64.b64encode(hmac.new(C.TURN_SECRET.encode(), user.encode(), hashlib.sha1).digest()).decode()
    return {"iceServers": [{"urls": f"stun:{C.TURN_HOST}:3478"},
                           {"urls": [f"turn:{C.TURN_HOST}:3478?transport=udp", f"turn:{C.TURN_HOST}:3478?transport=tcp"],
                            "username": user, "credential": key}]}


PUBLIC_SITE = "https://javis.bcsmethai.com"
ROBOTS_TXT = "\n".join([
    *(f"User-agent: {bot}\nAllow: /" for bot in (
        "Googlebot", "OAI-SearchBot", "ChatGPT-User", "GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended")),
    "User-agent: *",
    "Allow: /",
    "Disallow: /webrtc/",
    "Disallow: /outputs",
    "Disallow: /chat",
    "Disallow: /ice",
    "Disallow: /leads",
    f"Sitemap: {PUBLIC_SITE}/sitemap.xml",
    "",
])
SITEMAP_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    f"  <url><loc>{PUBLIC_SITE}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n"
    "</urlset>\n"
)
LLMS_TXT = (
    "# Javis — พนักงานตอบคำถามเสียงภาษาไทย BC Ai Account (ถามได้ทุกเรื่องในระบบ)\n\n"
    "> พนักงานตอบคำถามเสียงภาษาไทยสำหรับโปรแกรมบัญชี BC Ai Account (BC AI Account) ถามได้ทุกเรื่องในระบบ คุยโต้ตอบแบบเพื่อนผ่านเบราว์เซอร์ (WebRTC) หรือพิมพ์แชท "
    "ตอบคำถามการใช้งานโปรแกรมบัญชี BC Ai Account และระบบบัญชี SME ไทยจากสมองคลังความรู้บีซี พร้อมคำนวณตัวอย่างด้วย Decimal และรับฝากข้อความติดต่อกลับ\n\n"
    "## ความสามารถหลักบนสายสาธารณะ\n"
    "- ตอบคำถามได้ทุกเรื่องในระบบ ทั้งเมนู ขั้นตอนการใช้งาน และวิธีคิดต้นทุน/ภาษีในโปรแกรมบัญชี BC Ai Account จากสมองคลังความรู้บีซีโดยไม่เดาหรือแต่งข้อมูล\n"
    "- คำนวณตัวอย่างประกอบบนหน้าจอด้วยเครื่องคิดเลขทศนิยมแม่นยำ (Decimal ROUND_HALF_UP)\n"
    "- รับฝากชื่อ ร้าน/บริษัท เบอร์โทร และเรื่องที่ต้องการให้ติดต่อกลับ\n\n"
    "## ลิงก์หลัก\n"
    f"- [หน้าสนทนาหลัก]({PUBLIC_SITE}/): เปิดหน้าเว็บเพื่อเริ่มคุยด้วยเสียงหรือพิมพ์ข้อความได้ทันที\n"
)


def make_app(stream, public=False):
    tools = PUBLIC["tools"] if public else TOOLS

    @asynccontextmanager
    async def lifespan(_):
        task = None if public else asyncio.create_task(remind_loop(stream))
        yield
        task and task.cancel()

    app = FastAPI(lifespan=lifespan)
    stream.mount(app)  # POST /webrtc/offer

    @app.get("/")
    def index():
        return FileResponse(os.path.join(C.BASE_DIR, "static", "index.html"))

    @app.get("/team")  # who is on the team right now
    def team():
        return {"boss": C.DEEPSEEK_MODEL, "workers": sorted(set(WORKER.get(n, "other") for n in tools))}

    @app.get("/outputs")  # subtitles, status, info cards, panel (Server-Sent Events)
    async def outputs(webrtc_id: str):
        async def events():
            async for out in stream.output_stream(webrtc_id):
                yield f"data: {json.dumps(out.args[0], ensure_ascii=False)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/chat")  # typed message instead of speech; answered with voice + screen like a spoken turn
    async def post_chat(body: dict):
        call = str(body.get("webrtc_id"))
        text, handler = body.get("text"), stream.handlers.get(call)
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            return JSONResponse({"error": "text must be 1-500 characters"}, status_code=400)
        if not handler:
            return JSONResponse({"error": "no active call"}, status_code=409)
        PENDING[call]["text"] = f"{PENDING[call]['text']} {text.strip()}".strip()[:1000]
        if not handler.state.started_talking:  # typed while speaking: joins that turn
            handler.push_turn(interrupt=True)
        return {"ok": True}

    @app.get("/note")  # one BC note in full, behind a card or a [[link]] (ลุงจืด 2026-11-04 "กดดูได้ต่อเนื่อง")
    def note(slug: str):
        # ponytail: no call needed, the brain MCP itself is open to the internet; this reads one note by its slug
        if "read_note" not in tools or not re.fullmatch(r"[\w.-]{1,120}", slug):
            return JSONResponse({"error": "unknown note"}, status_code=404)
        try:
            text = tools["read_note"](slug=slug)[1]
        except Exception as e:
            log.warning("note %s failed: %s", slug, e)
            return JSONResponse({"error": "note unavailable"}, status_code=502)
        return {"text": (NOTE_META.sub("", text) if public else text)[:40000]}  # customers: content, not code paths

    @app.get("/robots.txt")
    def robots():
        return PlainTextResponse(ROBOTS_TXT)

    @app.get("/sitemap.xml")
    def sitemap():
        return Response(SITEMAP_XML, media_type="application/xml")

    @app.get("/llms.txt")
    def llms_txt():
        return PlainTextResponse(LLMS_TXT)

    if public:
        app.get("/ice")(ice_servers)
    else:  # customers' names and numbers: the owner line only, which listens on 127.0.0.1
        app.get("/leads")(lambda: HTMLResponse(leads_page()))

    return app


def tunnel():
    """Customers reach the public port through Caddy on PUBLIC_SSH: a reverse SSH tunnel, redialled when it drops.
    ponytail: a killed server can leave its ssh behind; that one still forwards to the same port, so it is harmless"""
    while True:
        subprocess.run(["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                        "-o", "ServerAliveCountMax=3", "-R", f"127.0.0.1:{C.PUBLIC_TUNNEL_PORT}:127.0.0.1:{C.PUBLIC_PORT}", C.PUBLIC_SSH],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.warning("public tunnel closed, redialling in 10 s")
        time.sleep(10)


async def serve():
    """The owner line and the public line: two ports, one process, so both share the models already on the GPU."""
    lines = [(make_app(make_stream()), C.SERVER_PORT), (make_app(make_stream(True), public=True), C.PUBLIC_PORT)]
    await asyncio.gather(*(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()
                           for app, port in lines))


if __name__ == "__main__":
    os.makedirs(C.DOCS_DIR, exist_ok=True)
    os.makedirs(C.SESSIONS_DIR, exist_ok=True)
    stt = load_stt()
    if not C.DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set: put it in .env (the boss runs on DeepSeek)")
    # without a timeout a hung Ollama hung the turn for good. 60 s covers warm_up's first model load (~25 s); a judge call
    # that times out is caught in judge() and cuts the piece
    llm = ollama.Client(host=C.OLLAMA_HOST, timeout=60)  # the judge (VISION_MODEL)
    tts = load_tts()
    load_mcp_tools()
    load_workers()
    log.info("STT=%s (cpu-onnx) BOSS=%s VISION=%s TTS=%s (%s)", C.STT_MODEL, C.DEEPSEEK_MODEL, C.VISION_MODEL, C.TTS_MODEL, C.DEVICE)
    warm_up()
    if C.PUBLIC_SSH:
        threading.Thread(target=tunnel, daemon=True).start()
    asyncio.run(serve())
