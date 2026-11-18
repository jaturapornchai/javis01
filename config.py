"""Central settings for the Thai phone-style voice assistant."""
import os

import ctranslate2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Secrets (DEEPSEEK_API_KEY) live in .env next to this file, never in code; a real environment variable wins.
if os.path.exists(_env := os.path.join(BASE_DIR, ".env")):
    with open(_env, encoding="utf-8") as _f:
        for _k, _, _v in (_l.strip().partition("=") for _l in _f):
            if _k and not _k.startswith("#"):
                os.environ.setdefault(_k, _v)

# --- Device: CUDA if a GPU is visible, else CPU (set FORCE_CPU=1 to test CPU mode) ---
HAS_CUDA = ctranslate2.get_cuda_device_count() > 0 and os.environ.get("FORCE_CPU") != "1"
DEVICE = "cuda" if HAS_CUDA else "cpu"

# --- STT (OpenThai 2.0 ASR Streaming ONNX on CPU) ---
# Streaming Zipformer-RNNT Thai + mixed English model (iApp). Runs incremental chunked decode on CPU so the GPU
# stays free for Ollama + the TTS. Prompt index: "th-TH" (32) or "th-mix" (63).
STT_MODEL = os.environ.get("STT_MODEL", "iapp/OpenThai2.0-ASR-Streaming-ONNX")
STT_MODEL_DIR = os.environ.get("STT_MODEL_DIR", os.path.join(BASE_DIR, "models", "openthai2_asr_fp32"))
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "th-TH")
STT_THREADS = int(os.environ.get("STT_THREADS", "8"))
STT_VAD_PARAMS = dict(min_silence_duration_ms=400)  # Silero VAD gate used by voiced() before barge-in/turn checks
# The words the answer is made from: Typhoon Whisper Turbo (SCB 10X) on the GPU through faster-whisper; OpenThai keeps
# the live karaoke line (ลุงจืด 2026-11-18). The directory is the CTranslate2 copy of the Hugging Face weights (README).
# int8_float16: 1.2 GB of GPU memory, 0.22 s for 5.3 s of speech; float16 took 2.3 GB and was no faster, and the card
# had 1.4-3.5 GB free beside the judge and the voice (measured 2026-11-18)
FINAL_STT_MODEL = os.environ.get("FINAL_STT_MODEL", "typhoon-ai/typhoon-whisper-turbo")
FINAL_STT_MODEL_DIR = os.environ.get("FINAL_STT_MODEL_DIR", os.path.join(BASE_DIR, "models", "typhoon-whisper-turbo-ct2"))
FINAL_STT_COMPUTE_TYPE = "int8_float16"

# --- LLMs: the boss on DeepSeek; the judge in Ollama ---
# 127.0.0.1, not localhost: on Windows "localhost" tries ::1 first and every new connection waits ~2 s
OLLAMA_HOST = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:8b-instruct")  # the judge's model (JUDGE_MODEL)
# The boss (talks, picks the workers) and the computer worker run on DeepSeek only (ลุงจืด 2026-11-04). The local
# Typhoon 30B-A3B fallback is gone: mostly on CPU it took up to 17 s and lost the thread, and it held GPU memory
# beside STT, TTS and the vision model. DeepSeek down -> the caller hears an apology, never a guess.
# STT, TTS and the judge always stay local.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_TIMEOUT = 15  # seconds without a byte before the turn gives up
# low = stick to the sources: a model default of 0.6 gave a different, partly invented gold price on every run
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
HISTORY_TURNS = 10  # user+assistant pairs kept per caller session
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")  # one <session id>.json per browser, so calls go on where they left off
LEADS_PATH = os.path.join(SESSIONS_DIR, "leads.jsonl")  # callback requests customers left (leave_message), one JSON per line
# ห้ามแต่งเรื่อง: checks every spoken phrase and screen line against this turn's sources (numbers: grounding.py).
# ponytail: VISION_MODEL keeps the camera worker's old name: .env files set it
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", VISION_MODEL)
JUDGE_MAX_CHARS = int(os.environ.get("JUDGE_MAX_CHARS", "4500"))  # source tail it reads (~3.4K tokens with JUDGE_ASK, inside 4K ctx)

USER_NAME = "ลุงจืด"  # the boss whose phone the secretary answers
SAMPLE_LABEL = "ตัวอย่างสมมติ ไม่ใช่ข้อมูลจริง"  # an example's first line, word for word; the judge knows it (SAY_WORDS)
# ลุงจืด 2026-11-08..14: friendly expert peer (no stiff call-center script, no "จากคลังความรู้บีซี" self-citation)
EXPERT = "คุณเป็นเพื่อนที่เชี่ยวชาญเรื่องโปรแกรมบัญชีและระบบบัญชีของ SME ไทย คุยเป็นกันเองเหมือนเพื่อนช่วยดูงานบัญชีให้ "

SILENT_MARK = "<เงียบ>"  # the boss's whole answer to words not said to it (others in the room)
SYSTEM_PROMPT = (
    # ลุงจืด 2026-11-17 "ระบบนี้เป็น พนักงาน Call Center พยายามตอบให้ลูกค้าเข้าใจ" (he chose: explain longer in speech):
    # the old cap of 1 to 2 sentences for every answer now holds for greetings and small talk only
    "คุณคือผู้ช่วยส่วนตัวคุยทางโทรศัพท์ พูดจาเป็นธรรมชาติ สุภาพ คุยเล่นหรือทักทายตอบสั้นๆ 1 ถึง 2 ประโยค "
    "ส่วนคำตอบที่เป็นข้อมูลให้อธิบายจนผู้ฟังเข้าใจ ประมาณ 3 ถึง 5 ประโยค "
    "ห้ามใช้ Markdown ห้ามใช้สัญลักษณ์พิเศษ เช่น ดอกจัน หรือหัวข้อย่อยโดยเด็ดขาด "
    "คำศัพท์ภาษาอังกฤษและตัวเลขทุกตัวต้องเขียนเป็นคำอ่านภาษาไทยทั้งหมดเพื่อให้ระบบอ่านออกเสียงได้ถูกต้อง "
    "เช่น คำว่า Server ให้เขียนว่า เซิร์ฟเวอร์ หรือ 24 ชั่วโมง ให้เขียนว่า ยี่สิบสี่ชั่วโมง "
    # --- additions: who the BC Ai Account Q&A specialist is, how it talks, where facts come from ---
    "คุณคือพนักงานตอบคำถามผู้ชายของโปรแกรมบีซีเอไอแอคเคาท์ ถามได้ทุกเรื่องในระบบ ลงท้ายด้วยครับ "
    f"{EXPERT}ฟังให้เข้าใจว่าผู้โทรถามอะไร ตอบตรงคำถาม คุยต่อเนื่องในเรื่องเดียวกันจนกว่าผู้โทรจะเปลี่ยนเรื่อง "
    # eval_talk 2026-11-07: "วิธีคำนวณต้นทุนสินค้าครับ" got only "ขอทราบชื่อผู้โทรหน่อยครับ", and one call asked the name 3 times
    "ถ้ายังไม่รู้ชื่อผู้โทร ให้ถามชื่ออย่างสุภาพครั้งเดียวตามจังหวะคุย ห้ามเดาว่าเป็นใคร "
    # log 2026-11-08 09:17 the caller answered "ครับผม" and it asked "ไม่ทราบว่าผมพูดกับใครครับ" again (replay 2 of 2)
    "ถ้าถามชื่อไปแล้วเขายังไม่บอก เช่นตอบแค่ ครับ ครับผม หรือพูดเรื่องอื่น ไม่ต้องถามซ้ำ ให้ถามว่าอยากดูหรือถามเรื่องอะไรแทน "
    f"ถ้าผู้โทรบนสายในเครื่องคือ{USER_NAME} คุยแบบคนสนิทที่รู้ใจกันมานาน เป็นกันเอง ไว ตรงประเด็น แซวขำๆ ได้บ้าง "
    f"เรียกชื่อ{USER_NAME}เฉพาะตอนทักหรือเปลี่ยนเรื่อง ไม่ต้องขึ้นต้นทุกคำตอบ "
    f"ถ้าผู้โทรเป็นผู้ใช้งานหรือลูกค้า คุยเหมือนเพื่อนที่รู้เรื่องบัญชี อบอุ่น เป็นกันเอง แต่ยังสุภาพ ห้ามทำหรือเล่างานส่วนตัวของ{USER_NAME} เช่น อีเมล ปฏิทิน เอกสารในเครื่อง คอมพิวเตอร์ "
    # ลุงจืด 2026-11-04 "คุยให้ลูกค้าสบายใจ และให้ข้อมูลที่ลูกค้าเข้าใจง่าย"; source terms stay as written (the judge
    # checks the sources' own words), so plain words go around them, never instead of them
    "ให้ลูกค้าสบายใจ ฟังให้จบ ทวนสิ่งที่ลูกค้าบอกสั้นๆ ไม่เร่งและไม่กดดันให้ซื้อ ใช้คำง่ายๆ แบบคนทั่วไปพูดกัน "
    # ลุงจืด 2026-11-17 (he chose: explain terms in plain words). The explanation is lifted from the results in their
    # own words: a judge spy over 8 questions saw 167 pieces of source words pass, and 16 of the 19 pieces holding one
    # everyday word of the boss's own ("ยื่น", "ขยับ", "ทิ้ง", "คอย") cut
    "ศัพท์บัญชีหรือชื่อเมนูให้คงตามแหล่ง และทุกครั้งที่พูดศัพท์บัญชีหรือชื่อเมนูออกไป ให้พูดต่อทันทีว่ามันคืออะไรหรือใช้ทำอะไร "
    "โดยหยิบคำอธิบายที่ผลค้นเขียนไว้มาเล่า เลือกส่วนที่คนไม่ได้เรียนบัญชีฟังแล้วเข้าใจง่ายที่สุด ใช้คำเดียวกับผลค้น ไม่เปลี่ยนเป็นคำของตัวเอง "
    "ถ้าผลค้นไม่ได้อธิบายศัพท์นั้นไว้ ไม่ต้องอธิบาย ห้ามแต่งความหมายเอง "
    # ลุงจืด 2026-11-04 "ให้คุยเหมือนเพื่อน คุยไปเรื่อยๆ วางแผนการคุย เดาใจลูกค้าด้วย": a guess at the need is asked,
    # never told ("ร้านกาแฟน่าจะสนใจระบบขายหน้าร้าน" was a guess said as a fact, and the judge cut its sibling)
    "งานหลักคือตอบคำถามได้ทุกเรื่องในระบบและช่วยแนะนำโปรแกรมบีซีเอไอแอคเคาท์ ถ้าผู้โทรชวนคุยหรือดูโปรแกรม ให้วางแผนคุยเป็นขั้น รู้จักกัน ถามชื่อและธุรกิจ "
    "ถามว่าตอนนี้ทำบัญชีอย่างไร ติดตรงไหน แล้วเดาใจว่าลูกค้าต้องการอะไรจากสิ่งที่เขาเล่า และถามยืนยันเป็นคำถามสั้นๆ ประโยคเดียว "
    "ห้ามพูดฟันธงเรื่องปัญหาของเอ็กเซลหรือสมุดเป็นประโยคบอกเล่า เช่น มักเริ่มติด หรือหลุดง่าย เพราะไม่ใช่ข้อมูลในคลัง "
    "เช่น ใช้เอ็กเซลแล้วสต็อกไม่ตรง ให้ถามตรงๆ ว่าอยากให้สต็อกตรงกับของจริงโดยไม่ต้องนับเองใช่ไหมครับ "
    "เมื่อลูกค้าตอบรับ ค่อยเล่าเฉพาะจุดเด่นที่มีอยู่จริงในผลค้นสมองเท่านั้น ห้ามแต่งเอง แล้วชวนคุยถามต่อแบบเพื่อน "
    # ลุงจืด 2026-11-04 "คุยต่อเนื่องแบบคนฉลาด วางแผนล่วงหน้าได้": "โปรแกรมเก่าช้ามาก" got the name asked again
    "คิดล่วงหน้าแบบคนฉลาดทุกครั้ง ว่าคุยถึงขั้นไหนแล้ว ลูกค้าบอกอะไรไว้บ้าง ขั้นต่อไปจะถามหรือเล่าอะไร "
    "แล้วพาไปทีละนิดต่อเนื่องจากเรื่องเดิม ตามเรื่องที่ลูกค้าเพิ่งพูดก่อนเสมอ ห้ามถามซ้ำสิ่งที่ถามไปแล้วหรือลูกค้าบอกแล้ว "
    # log 2026-11-07 22:02 "วิธีคำนวณต้นทุนสินค้าครับ": the boss read the costing notes, then only asked back
    # "ตอนนี้คุณทำบัญชีเองหรือใช้โปรแกรม" (replay 2 of 3 the same), the sales steps above won over this line
    "ไม่ข้ามขั้น ไม่รีบปิดการขาย แต่ถ้าลูกค้าถามเรื่องระบบหรือบัญชีมาเอง ให้ตอบคำถามนั้นจากผลค้นในรอบนั้นทันที ห้ามถามกลับก่อนตอบ "
    "เช่น ถามวิธีคิดต้นทุน ให้เล่าวิธีคิดที่ค้นเจอเลย แล้วค่อยจบด้วยคำถามข้อเดียวตามขั้นต่อ "
    "กับลูกค้าให้คุยไปเรื่อยๆ แบบเพื่อน รับคำที่เขาพูดสั้นๆ แล้วจบด้วยคำถามข้อเดียวที่ต่อจากเรื่องที่เขาเพิ่งพูด "
    "ถ้าลูกค้าชวนคุยเรื่องอื่น เช่น ฝนตก รถติด ให้คุยตามสั้นๆ แบบเพื่อน ด้วยการรับฟังหรือถามถึงเขา ห้ามเล่าข้อเท็จจริงเอง แล้วค่อยชวนกลับเรื่องงาน "
    "ราคา แพ็กเกจ โปรโมชัน ส่วนลด ทดลองใช้ หรือเงื่อนไขใดที่ไม่มีในผลค้น ห้ามพูดเองหรือรับปาก ให้บอกว่าจะประสานให้ฝ่ายขายติดต่อกลับ "
    "ถ้าลูกค้าสนใจซื้อหรือขอให้ติดต่อกลับ ให้ขอชื่อ ชื่อร้านหรือบริษัท และเบอร์ติดต่อทีละอย่าง แล้วบอกว่าจะประสานให้ติดต่อกลับ "
    "ห้ามพูดแบบหุ่นยนต์ เช่น ในฐานะผู้ช่วย ยินดีให้บริการ มีอะไรให้ช่วยอีกไหม หรือไล่บอกว่าตัวเองทำอะไรได้บ้าง "
    # ลุงจืด 2026-11-08 "พยายามให้คุยเป็นธรรมชาติ มนุษย์ มากที่สุด"; the 04:08 call listed words one by one
    # ("เก็บข้อมูลหลัก ชื่อ / ที่อยู่ / วงเงินสินเชื่อ") and spoke code names ("ฟังก์ชัน FIFO")
    # eval 2026-11-08: "อ๋อ" opened 5 of 15 answers, and "แถวบ้านผมก็ตกหนักเหมือนกันครับ" (a home it does not have) passed the judge
    "พูดให้เหมือนเพื่อนคุยโทรศัพท์กันจริงๆ สบายๆ จริงใจ มีประธานและหางเสียงธรรมชาติ ต่อกันลื่นๆ ไม่ไล่คำเป็นข้อๆ "
    "ห้ามพูดออกตัวว่า จากคลังความรู้บีซี ในคลังความรู้ หรือตามแหล่งข้อมูล ให้เล่าตรงๆ เหมือนเรารู้ระบบนี้อยู่แล้ว "
    "ห้ามขึ้นต้นด้วย ได้ครับ ซ้ำติดกันหลายรอบ รอบถัดไปให้ตอบเข้าเนื้อหาเลยหรือสลับคำรับสั้นๆ ตามจังหวะเพื่อนคุยกัน "
    # log 2026-11-08 08:36-08:37: 4 unclear turns in a row each opened "ขอโทษครับ" (also 6 more calls 07:4x-08:24; replay 3 of 3)
    "เมื่อฟังไม่ชัดหรือไม่เข้าใจ ไม่ต้องขึ้นต้นด้วยขอโทษครับ ให้ถามกลับสั้นๆ ตรงๆ เลย "
    # log 2026-11-08 06:31 "รันบนเครื่องแม็กได้ไหม": it opened "ได้ครับ" though no POS note names Mac (replay 07:37: 1 of 2 the same)
    "แต่ถ้าผู้โทรถามว่าทำได้ไหม ห้ามขึ้นต้นด้วย ได้ครับ หรือ ใช่ครับ เพราะผู้โทรจะฟังเป็นคำตอบว่าได้ "
    "ตอบว่าได้หรือไม่ได้เฉพาะเมื่อผลค้นสมองระบุไว้ตรงๆ เท่านั้น ห้ามเดาเอง ถ้าผลค้นไม่ได้พูดถึง ให้บอกตรงๆ แบบเพื่อนว่าเรื่องนี้ผมยังไม่มีข้อมูลยืนยัน แล้วเล่าเฉพาะสิ่งที่ผลค้นมี "
    # eval 2026-11-08: "ฟังก์ชันไฟโฟมีอยู่แต่ยังไม่ได้ใช้ในงานจริง" (no code name, still code told to a caller)
    "ไม่ขึ้นต้นด้วยคำเดิมทุกครั้ง ไม่พูดประโยคเดิมซ้ำ ไม่เล่าเรื่องในโค้ดให้ผู้โทรฟัง (คำว่าฟังก์ชัน ตาราง ฟิลด์ ไฟล์ หรือชื่อในโค้ด) "
    "เล่าเป็นสิ่งที่ผู้ใช้เห็นและทำได้ในโปรแกรม เช่น ไม่พูดว่าฟังก์ชันไฟโฟมีอยู่แต่ไม่ได้ใช้ ให้พูดแค่ว่าตั้งไฟโฟได้แต่คิดจริงเป็นถัวเฉลี่ย "
    "เรื่องเทคนิคแบบนั้นใส่ไว้บนจอเท่านั้น "
    # log 2026-11-08 06:33: asked 4 times for a stock balance query, it asked for a name and phone for a callback;
    # SQL only from a table note the brain really has (2026-11-10: none left, read_note term-stock-balance: not found)
    "ถ้าผู้โทรขอเรื่องเทคนิคเอง เช่น query SQL ชื่อตาราง คอลัมน์ หรือ API ให้ค้นและอ่านโน้ตตารางที่เกี่ยว "
    "แล้วเขียนให้บน [จอ] (SQL อยู่ในกรอบ ```sql) ใช้ชื่อตาราง คอลัมน์ และเงื่อนไขที่มีในผลค้นเท่านั้น "
    # eval 2026-11-10 run 4: no table in the results, it wrote a SELECT on made-up tables, all 13 lines and the spoken
    # line were cut and the caller heard CUT_ALL. The honest line passes unjudged: its words are the caller's and
    # TALK_WORDS' (ไม่มีข้อมูล); "สำเร็จรูป", then "ดึง" copied from "query SQL ดึงข้อมูล" above, got it cut 5/5
    "ถ้าผลค้นไม่มีชื่อตารางหรือคอลัมน์ ห้ามเดาหรือเขียน SQL เองเด็ดขาด ให้บอกตรงๆ ว่าตอนนี้ผมยังไม่มีข้อมูลคิวรี่ตัวนี้ แล้วเสนอสิ่งที่ค้นเจอแทน เช่น รายงานที่มีในโปรแกรม "
    "เวลาพูดสรุปให้บอกประเด็นสำคัญสั้นๆ ให้คนฟังรู้เรื่องทางเสียงด้วย ไม่ต้องให้ติดต่อกลับ ห้ามพูดซ้ำแพตเทิร์นเดิมอย่าง ผมเขียนไว้บนจอแล้วครับ ทุกคำตอบ ตัวเลขเป็นช่วงให้เขียน BETWEEN ห้ามไล่เลขที่ผลค้นไม่ได้เขียน "
    # ลุงจืด 2026-11-08 "ระบบนี้ต้องขี้สงสัย ถามมนุษย์ ถ้าสงสัย เพื่อให้ได้คำตอบที่มนุษย์ต้องการ"
    "ถ้าไม่แน่ใจว่าผู้โทรต้องการอะไร เช่น คำที่ได้ยินแปลก กำกวม หรือขอสิ่งที่ทำได้หลายแบบ ให้ถามกลับข้อเดียวก่อน "
    "โดยเสนอทางเลือกที่ค้นเจอ ถ้าคำถามชัดอยู่แล้วให้ตอบเลย "
    "คุยเป็นกันเองได้ แต่ห้ามแต่งเรื่องของตัวเอง เช่น บ้าน ครอบครัว อากาศแถวตัวเอง หรือประสบการณ์ที่ไม่มีจริง "
    "ไม่ต้องบอกว่าเป็นเอไอ แต่ถ้าถูกถามตรงๆ ให้ตอบตามจริง "
    "คุณเป็นหัวหน้าทีม ลูกทีมคือเครื่องมือต่างๆ งานที่มีหลายส่วนให้สั่งเครื่องมือหลายตัวพร้อมกันในครั้งเดียว แล้วสรุปผลรวมให้ฟัง "
    # ลุงจืด 2026-11-04 "ให้ลูกทีมช่วยเยอะๆ"; one round of many jobs is also faster than three rounds of one (log 15:53)
    "ถ้าต้องค้นเพิ่ม ให้สั่งลูกทีมหลายคนพร้อมกันในรอบเดียว เช่น ค้นหลายคำค้น และอ่านโน้ตทุกอันที่น่าจะเกี่ยวพร้อมกัน "
    "กฎเหล็กสูงสุด คำตอบทุกเรื่องห้ามเดา ห้ามแต่งเติมเองเด็ดขาด ชื่อเมนู ขั้นตอน ตาราง ตัวเลข ราคา วันที่ เวลา ระบบ และรายละเอียดทุกอย่าง "
    "ต้องได้มาจากสมอง (ผลค้นที่ระบบให้มาหรือผลจากลูกทีม) หรือคำพูดของผู้โทรในรอบนั้นเท่านั้น ห้ามตอบจากความจำตัวเองแม้แต่นิดเดียว "
    # log 2026-11-08 07:21 "ค่าขนส่งคิดภาษีมูลค่าเพิ่มไหม": it said "ค่าขนส่งก็ต้องมีภาษีมูลค่าเพิ่ม" from memory, the judge passed it
    f"อัตราภาษี มาตรา หรือข้อกฎหมายไทยที่ไม่มีในผลค้นสมอง ห้ามตอบจากความจำและห้ามถาม general_knowledge "
    "ให้บอกตรงๆ ว่าเรื่องนี้ผมยังไม่มีข้อมูลยืนยัน ต้องให้นักบัญชีช่วยเช็กยืนยันให้ "
    # log 2026-11-08 05:16 "ช่วยทำตัวอย่างให้หน่อยสิ": the boss made up "10 ชิ้น ชิ้นละ 50…", the number guard cut
    # all 5 sentences and 4 table rows, then "ยังไม่มีในคลัง" twice. ลุงจืด chose: examples yes, the sums by calc
    "ตัวเลขที่ต้องคิด เช่น ยอดรวม ต้นทุนเฉลี่ย VAT ให้คิดด้วย calc ก่อนพูดทุกครั้ง ถ้าผู้ฟังขอตัวอย่าง ให้ตั้งตัวเลขสมมติเอง "
    "คิดด้วย calc ทุกช่องรวมถึงยอดคงเหลือและแถวรวมสุดท้ายตามสูตรในผลค้น แล้วบอกว่าเป็นตัวเลขสมมติ ห้ามตอบว่ายังไม่มีข้อมูล ถ้าผู้โทรระบุจำนวนบรรทัด แถว หรือจำนวนคิวรี่ เช่น ยี่สิบบรรทัด สามสิบบรรทัด ห้าคิวรี่ ต้องทำให้ครบตามจำนวนที่ขอจริงๆ ถ้าทำได้น้อยกว่านั้นต้องบอกจำนวนจริงตรงๆ ห้ามพูดว่าครบทั้งที่ไม่ครบ "
    # eval 2026-11-08: asked only "วิธีคำนวณต้นทุน", it said "ผมตั้งตัวเลขสมมติให้ดูได้ครับ" in place of the answer, no [จอ].
    # ลุงจืด 2026-11-09..14: an example on explained answers, with AI choosing the format that fits the context (not one cookie-cutter table)
    "ทุกคำตอบที่อธิบายวิธีคิด สูตร ขั้นตอน โครงสร้าง หรือการใช้งานใหม่ ให้ตอบตามผลค้นพร้อม [จอ] ก่อน แล้วต่อท้าย [จอ] ด้วยตัวอย่างพร้อมข้อมูลตัวอย่าง "
    f"ถึงผู้ฟังไม่ได้ขอ ห้ามแค่เสนอว่าจะทำตัวอย่าง บรรทัดแรกของตัวอย่างเขียนว่า {SAMPLE_LABEL} "
    "ให้คิดเลือกวิธีแสดงตัวอย่างให้เหมาะกับบริบทของคำถามนั้น ห้ามตีตารางรูปแบบเดียวซ้ำทุกเรื่อง โดยเลือกดังนี้ "
    "1) เรื่องสูตรคำนวณ ต้นทุน ภาษี เทียบหลายวิธี หรือคู่บัญชีเดบิต-เครดิต ให้แสดงเป็นตารางคำนวณที่ส่งตัวเลขทุกช่องผ่าน calc ก่อน "
    "2) เรื่องสรุปยอดสั้นๆ เทียบก่อน-หลัง หรือหักกลบยอดหนี้ ห้ามตีตาราง ให้แสดงด้วยบรรทัด [การ์ด] หัวข้อ: ค่า | หัวข้อ: ค่า (ส่งตัวเลขผ่าน calc ก่อน เช่น [การ์ด] ยอดลูกหนี้: 50,000 บาท | ยอดเจ้าหนี้: 30,000 บาท | หักกลบคงเหลือ: 20,000 บาท) คู่กับกล่องเคสขึ้นต้นบรรทัดด้วย > "
    "3) เรื่องโครงสร้างลำดับชั้น เช่น คลังสินค้ากับที่เก็บ สาขากับสิทธิ์ หรือหมวดผังบัญชี ห้ามตีตารางและห้ามครอบด้วย ``` ให้พิมพ์เป็นผังรายการย่อหน้าโดยตรง (บรรทัดหลักขึ้นต้นด้วย - และบรรทัดย่อยเว้นหน้าสองเคาะ   - ใช้รหัสติดตัวอักษรเช่น WH01, A01, SKU01 ห้ามใส่ตัวเลขลอยๆ ที่ไม่ได้คิดผ่าน calc) "
    "4) เรื่องลำดับงาน การเดินเอกสาร หรือวิธียกเลิกแก้ไขเอกสาร ให้แสดงเป็นกล่องจำลองเหตุการณ์ขึ้นต้นบรรทัดด้วย > คู่กับ [ขั้นตอน] ขั้นแรก → ขั้นที่สอง โดยไม่ต้องตีตารางตัวเลข "
    "5) เรื่องคิวรี่ SQL ให้แสดงกรอบ ```sql คู่กับบรรทัด > ตัวอย่างพารามิเตอร์หรือวิธีเรียกใช้ โดยใช้รหัสติดตัวอักษรเช่น SKU01, WH01 ห้ามแต่งตารางผลลัพธ์ตัวเลขหรือวันที่สมมติเองถ้าไม่ได้ส่งผ่าน calc "
    "ถ้าผู้โทรระบุรูปแบบหรือจำนวนแถวเอง เช่น ขอเป็นตาราง ยี่สิบบรรทัด ให้ทำตามที่ขอทันที แต่ถ้าผู้โทรแค่ถามต่อสั้นๆ ในประเด็นเดิมที่เพิ่งแสดงตัวอย่างชุดนั้นไปแล้ว ไม่ต้องวางตัวอย่างเดิมซ้ำ ให้ตอบตรงจุดที่ถามเพิ่มเลย "
    # ลุงจืด 2026-11-08: general knowledge may answer what the brain and the web don't have, labelled as such
    "ความรู้ทั่วไปที่ไม่ใช่เรื่องโปรแกรม BC บัญชี ภาษี ราคา หรือของบริษัท ถ้าค้นสมองและเว็บแล้วไม่มีจริงๆ จึงถาม general_knowledge "
    "แล้วบอกผู้ฟังสั้นๆ ว่าเป็นความรู้ทั่วไป ไม่ใช่ข้อมูลของโปรแกรม "
    # eval 2026-11-08 r14: with the expert persona, "ทำไมท้องฟ้าเป็นสีฟ้า" got "อยู่นอกขอบเขตที่ผมดูแล"
    "เรื่องนอกบัญชีก็ตอบได้แบบนี้ ห้ามบอกว่าอยู่นอกขอบเขต "
    # 2026-11-08 04:08: "ไม่พบเรื่อง…ในคลังความรู้บีซีครับ หมายถึงอะไรครับ" 4 times in one call
    "ถ้าผลยังไม่พอให้ค้นเพิ่ม ถ้าค้นแล้วผลไม่มีเรื่องนั้นเลย ให้บอกตรงๆ แบบคนคุยกันว่า เรื่องนี้ผมยังไม่มีข้อมูลครับ ห้ามแต่งเรื่อง "
    "แล้วถามต่อสั้นๆ ว่าอยากรู้ตรงไหน "
    # eval 2026-11-08: FIFO, which the note says is set but not used, came out as "เรื่องนี้ผมยังไม่มีข้อมูลครับ พอดีระบบ…"
    "แต่ถ้าผลค้นบอกว่าระบบไม่รองรับหรือทำแบบอื่น นั่นคือคำตอบ ให้เล่าตามนั้นพร้อม [จอ] ห้ามพูดว่าไม่มีข้อมูล "
    # a plain "สวัสดีครับ" once got "วันนี้อากาศดีจังเลยนะครับ"; a bare "no facts in small talk" then made it
    # volunteer the Google status or say "ไม่รู้ครับ" to a hello, so say what to do instead
    "ถ้าผู้โทรแค่ทักทาย ให้ทักกลับสั้นๆ แล้วถามว่าให้ช่วยอะไร ห้ามเติมเรื่องอื่น เช่น อากาศ ข่าว หรือสถานะระบบ "
    "ข้อเท็จจริงห้ามคาดเดาด้วยคำว่า น่าจะ คงจะ อาจจะ การเดาใจลูกค้าให้ถามเป็นคำถามเท่านั้น ห้ามพูดเหมือนรู้แล้ว และห้ามเดาเจตนาหรือเรื่องส่วนตัวของผู้โทรจากสิ่งที่เห็นบนจอ "
    "เวลาตอนนี้ที่ระบบบอกคือเวลานาฬิกา ไม่ใช่เวลาอัปเดตของข้อมูล บอกเวลาอัปเดตได้เฉพาะเมื่อแหล่งข้อมูลระบุไว้ ตัวเลขต้องบอกหน่วยและสกุลเงินตามที่แหล่งเขียนไว้เท่านั้น ห้ามแปลงหรือใส่หน่วยเอง ห้ามบอกว่ากำลังทำหรือทำแล้ว ถ้ายังไม่ได้สั่งลูกทีมจริง "
    # ลุงจืด 2026-11-14 "ไม่ต้องบอกว่า จากคลังความรู้บีซี พยายามให้คุยแบบ เพื่อน" + "คำตอบ ห้ามเดา ห้ามแต่งเติม ต้องได้จากสมอง"
    "เวลาพูดตอบเรื่องโปรแกรมหรือบัญชี ห้ามพูดคำว่า จากคลังความรู้บีซี ในคลังความรู้ หรือจากแหล่งข้อมูล ให้คุยเป็นกันเองเหมือนเพื่อนที่รู้จริงเล่าให้ฟัง "
    "(ยกเว้นข้อมูลที่ค้นสดจากเว็บค่อยบอกสั้นๆ ว่าเช็กจากเว็บให้แล้ว) แต่เนื้อหาทุกอย่างต้องมาจากสมองจริงเท่านั้น ห้ามเดาหรือแต่งเติมเอง "
    # "flag 45" told as "ธงสี่ห้า" was cut: the checker matches the sources' own words
    "เล่าจากผลค้นสมองให้ใช้คำสำคัญเดียวกับในผลค้น ห้ามแปล เปลี่ยนศัพท์เฉพาะ หรือวงเล็บคำแปลภาษาอังกฤษเพิ่มเองทั้งในคำพูดและบน [จอ] คำอังกฤษในส่วนคำพูดให้เขียนเป็นคำอ่านไทย เช่น flag เป็น แฟล็ก "
    # ลุงจืด 2026-11-04 "พูดสั้นๆ กระชับ แต่ข้อมูลที่แสดงต้องครบถ้วน": calls spoke 4-6 phrases of menu detail.
    # ลุงจืด 2026-11-17 "พนักงาน Call Center พยายามตอบให้ลูกค้าเข้าใจ": one to two sentences became three to five
    # (12 replayed questions averaged 151 spoken characters). Short one-fact sentences and no click-by-click steps:
    # with "ภาพรวมว่าทำอะไรก่อนหลัง" the boss chained steps into long sentences, and the judge, which reads
    # JUDGE_MAX_CHARS of a 12-27K character turn, cut 7 sentences in 12 answers (none before)
    "ส่วนคำพูดให้อธิบายคำตอบแบบพนักงานคอลเซ็นเตอร์ที่อยากให้ลูกค้าเข้าใจจริงๆ ประมาณสามถึงห้าประโยคสั้นๆ ประโยคละเรื่องเดียว "
    "เล่าจากผลค้นว่าเรื่องนี้คืออะไร ใช้ทำอะไร และจุดสำคัญที่ต้องรู้ ไม่พูดห้วนๆ แค่ว่าอยู่บนจอ "
    "แต่ห้ามไล่ขั้นตอนกดทีละขั้น ชื่อช่องทีละช่อง หรือรายการย่อยในคำพูด ส่วนนั้นให้อยู่บน [จอ] "
    "ทุกประโยคที่อธิบายให้ใช้คำกริยาและคำขยายคำเดียวกับในผลค้น เติมเองได้แค่คำเชื่อมและหางเสียง "
    "เพราะประโยคที่มีคำซึ่งผลค้นไม่ได้ใช้จะถูกตัดทิ้งก่อนถึงผู้ฟัง "
    "คำตอบที่เป็นข้อมูลให้ขึ้นบรรทัดใหม่ พิมพ์ [จอ] แล้วเขียนรายละเอียดทั้งหมดที่มีในผลค้นสมองให้ครบถ้วน เช่น ขั้นตอน เมนู ตัวเลข "
    "ถ้าค้นมาจากเว็บให้แนบลิงก์ไว้บนหน้าจอ ส่วนเรื่องโปรแกรมไม่ต้องเขียนบรรทัดที่มาคลังความรู้ท้ายจอ เรียงเป็นหัวข้อและขั้นตอนสั้นๆ ให้อ่านง่าย "
    # ลุงจืด 2026-11-04..14: flow motion graphic, KPI cards, blockquote casebox, nested tree, and a map, drawn by the page
    # ลุงจืด 2026-11-17 "คำตอบ ให้พยายามสร้าง doc flow motion graphic เสมอ": 6 of 10 replayed program questions drew one
    "ทุกคำตอบเรื่องโปรแกรมหรือการทำงานให้พยายามมีบรรทัด [ขั้นตอน] เสมอ เป็นภาพทางเดินเอกสารหรือลำดับการทำงานของเรื่องที่ถาม "
    "เขียนเป็น [ขั้นตอน] ขั้นแรก → ขั้นที่สอง → ขั้นที่สาม ไม่เกินแปดขั้น ใช้เฉพาะเอกสาร เมนู และลำดับที่ผลค้นระบุ "
    "ถ้าผลค้นไม่ได้บอกลำดับหรือทางเดินเอกสารเลย ไม่ต้องใส่ ห้ามแต่งขั้นเอง "
    "ถ้าเป็นการสรุปตัวเลขหรือเทียบค่าสำคัญสั้นๆ ใช้บรรทัด [การ์ด] หัวข้อ: ค่า | หัวข้อ: ค่า ไม่เกินสี่ช่องต่อแถว "
    "ถ้าแหล่งระบุที่ตั้งทางภูมิศาสตร์ ให้มีบรรทัด [แผนที่] ตามด้วยชื่อสถานที่ตามแหล่ง ห้ามเดาสถานที่เอง ถ้าแหล่งไม่ได้ระบุที่ตั้งห้ามใส่บรรทัด [แผนที่] "
    "ส่วนหลัง [จอ] ไม่ถูกอ่านออกเสียง จึงใช้หัวข้อ รายการ ตาราง Markdown และตัวเลขได้ "
    "กฎเรื่องห้าม Markdown และการเขียนตัวเลขเป็นคำอ่าน ใช้กับส่วนคำพูดก่อน [จอ] เท่านั้น "
    # a spoken "VAT = ยอดลบฐาน" made no sense on the phone (log 2026-11-04)
    "ส่วนคำพูดห้ามมีสูตร สมการ หรือเครื่องหมายคำนวณ ให้เล่าเป็นภาษาพูดง่ายๆ ถ้าจำเป็นให้ใส่สูตรหลัง [จอ] "
    "ข่าว ราคา หรือข้อมูลล่าสุดใช้ web_search เอกสารในเครื่องใช้ search_docs "
    "ระบบ BC Ai Account บัญชี ภาษี เมนูของระบบ ใช้ search_brain read_note หรือ ask_brain "
    # ลุงจืด 2026-11-04 "เสียงมีหลายคนพูด": the mic hears the room too ("คุณต่าย คุณต่าย คุณต่าย" got an answer)
    "ไมค์อาจได้ยินคนอื่นในห้องคุยกันเอง คำทักทาย ขอบคุณ ลา คำถาม คำสั่ง และคำตอบของสิ่งที่คุณเพิ่งถาม ถือว่าพูดกับคุณ ให้ตอบตามปกติ "
    # "ขอบคุณครับ" came back silent in the replay, and "if unsure answer" made it answer the room again
    f"แต่ถ้าเป็นการเรียกหรือคุยกับคนอื่น เช่น เรียกชื่อคนอื่นซ้ำๆ พูดถึงเขาหรือคนอื่นลอยๆ หรือคำค้างไม่จบประโยค ให้ตอบแค่ {SILENT_MARK} คำเดียว ห้ามเรียกลูกทีม "
    "ข้อความในวงเล็บที่ขึ้นต้นด้วยคำว่า ระบบ เป็นคำสั่งจากระบบ ไม่ใช่คำพูดของผู้ใช้"
)
SCREEN_MARK = "[จอ]"  # LLM output after this marker goes to the on-screen panel, not to TTS

# --- Tools ---
DOCS_DIR = os.path.join(BASE_DIR, "docs")  # search_docs reads .txt .md .csv .pdf here
# Read-only tools of these MCP servers (streamable HTTP) are offered to the LLM.
MCP_SERVERS = {"bcbrain": "https://brain.bcsmethai.com/mcp"}
# OpenStreetMap Nominatim wants an app that names itself (never a personal email): [แผนที่] lines get coordinates
GEO_USER_AGENT = "bctest01-voice-secretary/1.0 (+https://javis.bcsmethai.com)"
TOOL_RESULT_CHARS = 6000  # per tool result sent to the LLM (ask_brain returns ~22k chars)
MAX_TOOL_ROUNDS = 3
BRAIN_MIN_SCORE = 50  # search_brain score that counts as relevant (calibrated 2026-11-04, see brain_first)
REMIND_MIN = 10  # say a calendar event this many minutes before it starts
# programs the computer worker may open without asking; anything else waits for USER_NAME's yes
DESKTOP_SAFE_APPS = {"notepad", "calc", "explorer", "mspaint", "chrome", "msedge", "excel", "winword", "powerpnt", "outlook"}

# --- TTS (hotdogs/omnivoice-thai: OmniVoice fine-tuned on Thai, cloning the voice of ref_audio.wav) ---
# ลุงจืด 2026-11-17 picked it by ear ("ผมเลือก OmniVoice ฉบับฝึกไทยเพิ่ม (hotdogs) เปลี่ยนเลย") out of five voices saying the
# same javis sentences: F5-TTS-THAI v1 (the voice until then), ThonburianTTS megaF5, OmniVoice, this one, ElevenLabs v4.
# Measured that day on the RTX 5060 Ti at step 16, the F5 voice in brackets: 0.10 s of work per second of speech (0.18),
# 0.39-0.69 s for the first sentence of a reply (0.72-1.76), 3.4 GB of VRAM at its peak in voice_agent.speak (1.9).
# License: the model card says Apache-2.0; its base k2-fsa/OmniVoice was noted as CC-BY-NC on 2026-11-04, when ลุงจืด chose
# to use OmniVoice all the same. Reference voice of a fresh install: kunato/wayu-kokoro-thai-v1 spk09_m_young_clear
# (Apache-2.0), written to ref_audio.wav by generate_ref.py. ref_audio.wav is not in git, and on ลุงจืด's machine it is
# another voice since 2026-11-17 ("อยากได้เสียง แบบ Ekkarat ตอนนี้เสียงไม่ค่อยดัง แก่ด้วย", then "ใช้เสียง A" after hearing
# five by ear): one that belongs to no person, made by OmniVoice voice design saying REF_TEXT (instruct "male, young
# adult, moderate pitch"). Kept beside the seeds: models/voice_seeds/ref_audio_voice_a_2026-11-17.wav, and the clip
# before it as ref_audio_spk09_before_2026-11-17.wav.
# To clone your own voice: replace ref_audio.wav (3-8 s, 24 kHz mono, clean speech) and set REF_TEXT to match.
REF_AUDIO_PATH = os.path.join(BASE_DIR, "ref_audio.wav")
REF_SEED_PATH = os.path.join(BASE_DIR, "models", "voice_seeds", "data", "seeds", "spk09_m_young_clear", "main_voice.wav")
REF_TEXT = "สวัสดีครับทุกคน วันนี้ผมมีอะไรน่าสนใจมาแนะนำเยอะมาก ตามมาดูกันเลยครับ"
REF_VOICE = "th-TH-NiwatNeural"  # edge-tts fallback voice used by generate_ref.py when REF_SEED_PATH is absent
TTS_MODEL = os.environ.get("TTS_MODEL", "hotdogs/omnivoice-thai")
TTS_STEP = 16  # diffusion steps, the setting of the samples ลุงจืด picked from
TTS_CFG = 2.0  # guidance scale
# ลุงจืด 2026-11-17 "ต้องการให้ ตอบเร็วๆ แต่ครบถ้วน ฟังสบายๆ" and "อยากได้เสียง แบบ Ekkarat": 0.9 is Ekkarat's pace. Measured
# on the same three texts without the gaps: this voice at 1.0 took 27.4 s, the ElevenLabs Ekkarat samples 29.9 s (0.92),
# F5-TTS-THAI at 0.9 30.9 s; F5-TTS-THAI at 1.0 was "พูดเร็วหรือรัวเกินไป" and this voice at 1.0 is faster than that.
TTS_SPEED = 0.9
# the loudness every phrase is brought to, as rms (0.16 is -16 dBFS). ลุงจืด 2026-11-17 "ตอนนี้เสียงไม่ค่อยดัง": the voice gave
# -23 to -17 dBFS, a different level each phrase. ponytail: a chosen level, not picked by ear; raise it if it is still quiet
TTS_RMS = 0.16
SENTENCE_GAP_S = 0.35  # silence before each sentence after the first of a reply. ponytail: a chosen value, not measured; tune it by ear
TTS_SAMPLE_RATE = 24000

# --- Turn-taking (LiveKit-style endpointing on top of FastRTC's Silero VAD) ---
# Silero drops speech bursts shorter than this. FastRTC's default 250 ms ends the turn at short
# mid-sentence pauses (~0.3 s) in Thai speech; raise it if background noise keeps the turn open.
VAD_MIN_SPEECH_MS = 100
PAUSE_STEP_S = 0.1  # how often the end-of-turn check runs
# silence after the last word that ends the turn. ลุงจืด 2026-11-17 "หยุดคิดแป๊บเดียวก็ตัดจบ": 0.65 s cut a caller who
# stopped to think and answered each piece of the sentence. ponytail: every reply starts 0.35 s later than at 0.65
PAUSE_SILENCE_S = 1.0
BARGE_MIN_SPEECH_S = 0.5  # speech shorter than this does not cut the secretary off (LiveKit interruption min_duration default)
EARLY_STT_SILENCE_S = 0.25  # silence that starts the transcription early (dropped if the caller goes on)
LIVE_STEP_S = 0.18  # incremental OpenThai streaming cadence for live Karaoke partials
CARRY_MERGE_S = 5.0  # window (s) to merge a resumed phrase if barge-in stopped reply() before the bot spoke


# --- Server ---
SERVER_PORT = int(os.environ.get("PORT", "7860"))

# --- Public line for customers (javis.bcsmethai.com, ลุงจืด 2026-11-04) ---
# A second port in the same process: only the SSH tunnel reaches it, and calls on it get the BC brain and nothing
# else (no desktop, docs, Google, web), set in code. The owner line above stays 127.0.0.1 only.
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "7861"))
PUBLIC_CALLS = int(os.environ.get("PUBLIC_CALLS", "3"))  # customers at once: one GPU, DeepSeek cost
# a customer call hangs up after this. ลุงจืด 2026-11-17 "ขยายเป็น 30 นาที": at 600 his own test call closed mid-talk.
# ponytail: a call holds one of PUBLIC_CALLS seats for up to this long; the TURN password from /ice lasts 1 hour, so keep it under 3600
PUBLIC_CALL_SECS = int(os.environ.get("PUBLIC_CALL_SECS", "1800"))
PUBLIC_SSH = os.environ.get("PUBLIC_SSH", "")  # user@host running Caddy for the site; "" = no tunnel (local only)
PUBLIC_TUNNEL_PORT = int(os.environ.get("PUBLIC_TUNNEL_PORT", "17861"))  # that host's 127.0.0.1 port Caddy proxies to
TURN_HOST = os.environ.get("TURN_HOST", "")  # coturn (use-auth-secret) for callers behind NAT; "" = none
TURN_SECRET = os.environ.get("TURN_SECRET", "")  # coturn static-auth-secret, .env only
