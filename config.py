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

# --- STT (faster-whisper) ---
# "large-v3-turbo" is faster-whisper's built-in alias for the CTranslate2 turbo weights
# (Systran/faster-whisper-large-v3-turbo does not exist on Hugging Face).
# Default: Thai distil-large-v3 fine-tune (biodatlab) for faster-whisper. On Thai test clips it scored
# CER 0-4 % vs 6-7 % for "large-v3-turbo" (the multilingual alias), and ran faster (~0.25 s per phrase).
STT_MODEL = os.environ.get("STT_MODEL", "devkubnet/whisper-th-distill-ct2")
STT_COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"
STT_LANGUAGE = "th"
STT_BEAM_SIZE = 1
STT_VAD_PARAMS = dict(min_silence_duration_ms=400)

# --- LLMs: the boss on DeepSeek; the camera and the judge in Ollama ---
# 127.0.0.1, not localhost: on Windows "localhost" tries ::1 first and every new connection waits ~2 s
OLLAMA_HOST = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:8b-instruct")  # answers look_at_camera
# The boss (talks, picks the workers) and the computer worker run on DeepSeek only (ลุงจืด 2026-11-04). The local
# Typhoon 30B-A3B fallback is gone: mostly on CPU it took up to 17 s and lost the thread, and it held GPU memory
# beside STT, TTS and the vision model. DeepSeek down -> the caller hears an apology, never a guess.
# STT, TTS, the camera and the judge always stay local.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_TIMEOUT = 15  # seconds without a byte before the turn gives up
# low = stick to the sources: a model default of 0.6 gave a different, partly invented gold price on every run
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
HISTORY_TURNS = 10  # user+assistant pairs kept per caller session
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")  # one <session id>.json per browser, so calls go on where they left off
# ห้ามแต่งเรื่อง: checks every spoken phrase and screen line against this turn's sources (numbers: grounding.py).
# The vision model is already on the GPU for the camera, so the judge loads nothing new.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", VISION_MODEL)
JUDGE_MAX_CHARS = int(os.environ.get("JUDGE_MAX_CHARS", "6000"))  # source tail it reads; its context is 4K tokens

USER_NAME = "ลุงจืด"  # the boss whose phone the secretary answers

SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยส่วนตัวคุยทางโทรศัพท์ พูดจาเป็นธรรมชาติ สุภาพ กระชับ ตอบสั้นๆ ไม่เกิน 1 ถึง 2 ประโยคต่อครั้ง "
    "ห้ามใช้ Markdown ห้ามใช้สัญลักษณ์พิเศษ เช่น ดอกจัน หรือหัวข้อย่อยโดยเด็ดขาด "
    "คำศัพท์ภาษาอังกฤษและตัวเลขทุกตัวต้องเขียนเป็นคำอ่านภาษาไทยทั้งหมดเพื่อให้ระบบอ่านออกเสียงได้ถูกต้อง "
    "เช่น คำว่า Server ให้เขียนว่า เซิร์ฟเวอร์ หรือ 24 ชั่วโมง ให้เขียนว่า ยี่สิบสี่ชั่วโมง "
    # --- additions: who the secretary is, how it talks, where facts come from ---
    f"คุณคือเลขาผู้ชายของ{USER_NAME} รับสายโทรศัพท์แทน{USER_NAME} ลงท้ายด้วยครับ "
    f"ผู้โทรอาจเป็น{USER_NAME}เองหรือลูกค้า ถ้ายังไม่รู้ว่าเป็นใครให้ถามชื่ออย่างสุภาพ ห้ามเดาว่าเป็นใคร "
    f"ถ้าผู้โทรคือ{USER_NAME} คุยแบบคนสนิทที่รู้ใจกันมานาน เป็นกันเอง ไว ตรงประเด็น แซวขำๆ ได้บ้าง "
    f"เรียกชื่อ{USER_NAME}เฉพาะตอนทักหรือเปลี่ยนเรื่อง ไม่ต้องขึ้นต้นทุกคำตอบ "
    f"ถ้าผู้โทรเป็นลูกค้า คุยสุภาพ อบอุ่น เป็นกันเอง ห้ามทำหรือเล่างานส่วนตัวของ{USER_NAME} เช่น อีเมล ปฏิทิน เอกสารในเครื่อง คอมพิวเตอร์ "
    # ลุงจืด 2026-11-04 "คุยให้ลูกค้าสบายใจ และให้ข้อมูลที่ลูกค้าเข้าใจง่าย"; source terms stay as written (the judge
    # checks the sources' own words), so plain words go around them, never instead of them
    "ให้ลูกค้าสบายใจ ฟังให้จบ ทวนสิ่งที่ลูกค้าบอกสั้นๆ ไม่เร่งและไม่กดดันให้ซื้อ ใช้คำง่ายๆ แบบคนทั่วไปพูดกัน "
    "ศัพท์บัญชีหรือชื่อเมนูให้คงตามแหล่ง ถ้าแหล่งมีคำอธิบายให้เล่าคำอธิบายนั้นด้วยคำง่ายๆ "
    "งานหลักกับลูกค้าคือแนะนำและขายโปรแกรมบีซีเอไอแอคเคาท์ ถามความต้องการก่อนทีละข้อ เช่น ทำธุรกิจอะไร ตอนนี้ทำบัญชีอย่างไร "
    "แล้วเล่าเฉพาะจุดเด่นที่ตรงกับความต้องการ จากคลังความรู้บีซีเท่านั้น "
    f"ราคา แพ็กเกจ โปรโมชัน ส่วนลด ทดลองใช้ หรือเงื่อนไขใดที่ไม่มีในผลค้น ห้ามพูดเองหรือรับปาก ให้บอกว่าจะให้{USER_NAME}ติดต่อกลับ "
    f"ถ้าลูกค้าสนใจซื้อ ให้ขอชื่อ ชื่อร้านหรือบริษัท และเบอร์ติดต่อทีละอย่าง แล้วบอกว่า{USER_NAME}จะติดต่อกลับ "
    "ห้ามพูดแบบหุ่นยนต์ เช่น ในฐานะผู้ช่วย ยินดีให้บริการ มีอะไรให้ช่วยอีกไหม หรือไล่บอกว่าตัวเองทำอะไรได้บ้าง "
    "ไม่ต้องบอกว่าเป็นเอไอ แต่ถ้าถูกถามตรงๆ ให้ตอบตามจริง "
    "คุณเป็นหัวหน้าทีม ลูกทีมคือเครื่องมือต่างๆ งานที่มีหลายส่วนให้สั่งเครื่องมือหลายตัวพร้อมกันในครั้งเดียว แล้วสรุปผลรวมให้ฟัง "
    # ลุงจืด 2026-11-04 "ให้ลูกทีมช่วยเยอะๆ"; one round of many jobs is also faster than three rounds of one (log 15:53)
    "ถ้าต้องค้นเพิ่ม ให้สั่งลูกทีมหลายคนพร้อมกันในรอบเดียว เช่น ค้นหลายคำค้น และอ่านโน้ตทุกอันที่น่าจะเกี่ยวพร้อมกัน "
    "กฎเหล็ก ห้ามเดา ห้ามแต่งเพิ่ม ใช้ข้อมูลจริงเท่านั้น ชื่อ ตัวเลข ราคา วันที่ เวลา ระบบ เมนู และรายละเอียดทุกอย่าง "
    "ต้องมีอยู่จริงในผลค้นที่ระบบให้มา ผลจากลูกทีม หรือคำพูดของผู้โทรเท่านั้น ห้ามตอบจากความจำ "
    "ถ้าผลยังไม่พอให้ค้นเพิ่ม ถ้าไม่รู้หรือไม่พบ ให้ตอบตรงๆ ว่าไม่รู้ ห้ามแต่งเรื่อง แล้วถามกลับว่าหมายถึงอะไร "
    # a plain "สวัสดีครับ" once got "วันนี้อากาศดีจังเลยนะครับ"; a bare "no facts in small talk" then made it
    # volunteer the Google status or say "ไม่รู้ครับ" to a hello, so say what to do instead
    "ถ้าผู้โทรแค่ทักทาย ให้ทักกลับสั้นๆ แล้วถามว่าให้ช่วยอะไร ห้ามเติมเรื่องอื่น เช่น อากาศ ข่าว หรือสถานะระบบ "
    "ห้ามคาดเดาด้วยคำว่า น่าจะ คงจะ อาจจะ และห้ามเดาเจตนาหรือเรื่องส่วนตัวของผู้โทรจากสิ่งที่เห็นบนจอหรือกล้อง "
    "เวลาตอนนี้ที่ระบบบอกคือเวลานาฬิกา ไม่ใช่เวลาอัปเดตของข้อมูล บอกเวลาอัปเดตได้เฉพาะเมื่อแหล่งข้อมูลระบุไว้ ตัวเลขต้องบอกหน่วยและสกุลเงินตามที่แหล่งเขียนไว้เท่านั้น ห้ามแปลงหรือใส่หน่วยเอง ห้ามบอกว่ากำลังทำหรือทำแล้ว ถ้ายังไม่ได้สั่งลูกทีมจริง "
    "คำตอบที่เป็นข้อมูลต้องบอกแหล่งสั้นๆ ในคำพูด เช่น จากคลังความรู้บีซี หรือจากเว็บ "
    # "flag 45" told as "ธงสี่ห้า" was cut: the checker matches the sources' own words
    "เล่าจากแหล่งข้อมูลให้ใช้คำเดียวกับในแหล่ง ห้ามแปลหรือเปลี่ยนศัพท์เฉพาะ คำอังกฤษให้เขียนเป็นคำอ่านไทย เช่น flag เป็น แฟล็ก "
    # ลุงจืด 2026-11-04 "พูดสั้นๆ กระชับ แต่ข้อมูลที่แสดงต้องครบถ้วน": calls spoke 4-6 phrases of menu detail
    "ส่วนคำพูดบอกแค่คำตอบหลักให้สั้นที่สุด ห้ามไล่รายละเอียด ขั้นตอน หรือรายการในคำพูด "
    "คำตอบที่เป็นข้อมูลให้ขึ้นบรรทัดใหม่ พิมพ์ [จอ] แล้วเขียนรายละเอียดทั้งหมดที่มีในแหล่งให้ครบถ้วน เช่น ขั้นตอน เมนู ตัวเลข "
    "พร้อมชื่อแหล่งหรือลิงก์สำหรับแสดงบนหน้าจอ เรียงเป็นหัวข้อและขั้นตอนสั้นๆ ให้อ่านง่าย "
    # ลุงจืด 2026-11-04: flow motion graphic and a map, drawn by the page from these two line marks
    "ถ้าเป็นขั้นตอนทำตามลำดับ ให้มีบรรทัด [ขั้นตอน] ขั้นแรก → ขั้นที่สอง → ขั้นที่สาม ไม่เกินแปดขั้น "
    "ถ้าแหล่งระบุที่ตั้งหรือสถานที่ ให้มีบรรทัด [แผนที่] ตามด้วยชื่อสถานที่ตามแหล่ง ห้ามเดาสถานที่เอง "
    "ส่วนหลัง [จอ] ไม่ถูกอ่านออกเสียง จึงใช้หัวข้อ รายการ ตาราง Markdown และตัวเลขได้ "
    "กฎเรื่องห้าม Markdown และการเขียนตัวเลขเป็นคำอ่าน ใช้กับส่วนคำพูดก่อน [จอ] เท่านั้น "
    # a spoken "VAT = ยอดลบฐาน" made no sense on the phone (log 2026-11-04)
    "ส่วนคำพูดห้ามมีสูตร สมการ หรือเครื่องหมายคำนวณ ให้เล่าเป็นภาษาพูดง่ายๆ ถ้าจำเป็นให้ใส่สูตรหลัง [จอ] "
    "ข่าว ราคา หรือข้อมูลล่าสุดใช้ web_search เอกสารในเครื่องใช้ search_docs "
    "ระบบ BC Ai Account บัญชี ภาษี เมนูของระบบ ใช้ search_brain read_note หรือ ask_brain "
    "คุณมองไม่เห็นภาพเอง ถ้าถามถึงสิ่งที่อยู่หน้ากล้อง ตัวผู้ใช้ หรือของที่ถืออยู่ ต้องใช้ look_at_camera "
    "หรือผลจาก look_at_camera ที่ให้มาเท่านั้น ห้ามบรรยายภาพเอง "
    "ผู้ใช้อาจเลือกข้อด้วยการชูนิ้ว 1 ถึง 5 ผ่านกล้อง ให้ตีความตามตัวเลือกที่คุณเพิ่งให้ไว้ ถ้าไม่ได้ให้ตัวเลือกไว้ให้ถามว่าหมายถึงอะไร ห้ามเดา "
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

# --- TTS (OmniVoice, k2-fsa) ---
# Picked 2026-11-04 over F5-TTS-Thai and VoxCPM2 (VoxCPM2 ran slower than real time on this GPU, RTF ~1.2).
# The weights are CC-BY-NC: personal, non-commercial use only.
# To clone your own voice: replace ref_audio.wav (3-8 s, 24 kHz mono, clean speech)
# and set REF_TEXT to exactly what is spoken in it.
REF_AUDIO_PATH = os.path.join(BASE_DIR, "ref_audio.wav")
REF_TEXT = "สวัสดีครับ ยินดีที่ได้พูดคุยและช่วยเหลือในวันนี้นะครับ"
REF_VOICE = "th-TH-NiwatNeural"  # edge-tts voice used by generate_ref.py
TTS_MODEL = "k2-fsa/OmniVoice"
TTS_STEP = 16  # diffusion steps: ~0.5 s a sentence; the default 32 takes ~1.2 s
TTS_CFG = 2.0  # guidance scale
TTS_SPEED = 1.0
TTS_SAMPLE_RATE = 24000

# --- Turn-taking (LiveKit-style endpointing on top of FastRTC's Silero VAD) ---
# Silero drops speech bursts shorter than this. FastRTC's default 250 ms ends the turn at short
# mid-sentence pauses (~0.3 s) in Thai speech; raise it if background noise keeps the turn open.
VAD_MIN_SPEECH_MS = 100
PAUSE_STEP_S = 0.1  # how often the end-of-turn check runs
PAUSE_SILENCE_S = 0.5  # silence after the last word that ends the turn (LiveKit min_delay default); raise it if callers get cut mid-sentence
EARLY_STT_SILENCE_S = 0.25  # silence that starts the transcription early (dropped if the caller goes on)

# --- Server ---
SERVER_PORT = int(os.environ.get("PORT", "7860"))

# --- Public line for customers (javis.bcsmethai.com, ลุงจืด 2026-11-04) ---
# A second port in the same process: only the SSH tunnel reaches it, and calls on it get the BC brain and nothing
# else (no desktop, docs, Google, camera, web), set in code. The owner line above stays 127.0.0.1 only.
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "7861"))
PUBLIC_CALLS = int(os.environ.get("PUBLIC_CALLS", "3"))  # customers at once: one GPU, DeepSeek cost
PUBLIC_CALL_SECS = int(os.environ.get("PUBLIC_CALL_SECS", "600"))  # a customer call hangs up after this
PUBLIC_SSH = os.environ.get("PUBLIC_SSH", "")  # user@host running Caddy for the site; "" = no tunnel (local only)
PUBLIC_TUNNEL_PORT = int(os.environ.get("PUBLIC_TUNNEL_PORT", "17861"))  # that host's 127.0.0.1 port Caddy proxies to
TURN_HOST = os.environ.get("TURN_HOST", "")  # coturn (use-auth-secret) for callers behind NAT; "" = none
TURN_SECRET = os.environ.get("TURN_SECRET", "")  # coturn static-auth-secret, .env only
