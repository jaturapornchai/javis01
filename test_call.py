"""End-to-end check without a mic: a real WebRTC call (aiortc client) into the same app voice_agent serves.

Turn 0 is the secretary's own greeting when the call connects, turn 1 asks about a local document (search_docs),
turn 2 asks for a sum and is talked over (barge-in), turn 3 is typed chat about BC (knowledge base first). Subtitles/cards/panel/team come over SSE.
Bot audio is saved to out_call.wav. Stop voice_agent.py first (this loads its own models).
Run: python test_call.py
"""
import asyncio
import base64
import fractions
import json
import os
import tempfile
import threading
import time
import uuid

import edge_tts
import httpx
import librosa
import numpy as np
import ollama
import soundfile as sf
import uvicorn
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

import config as C
import voice_agent as va

# --- pure logic --------------------------------------------------------------------------------------------------
chunks = list(va.chunk_sentences(iter(["สวัสดีครับ ", "วันนี้อากาศ", "ดีมาก ", "ราคา 1,250.", "50 บาทครับ"])))
assert "".join(chunks) == "สวัสดีครับ วันนี้อากาศดีมาก ราคา 1,250.50 บาทครับ", chunks
assert not any(c.endswith("1,250.") for c in chunks), chunks
phone = list(va.chunk_sentences(iter(["รับทราบครับ ร้านสมหญิงวัสดุ ", "เบอร์ศูนย์แปด", "หนึ่ง ", "หนึ่งสองสาม ส", "ี่ห้าหกเจ็ด ", "นะครับ"])))
assert any("ศูนย์แปดหนึ่ง หนึ่งสองสาม สี่ห้าหกเจ็ด" in c for c in phone), phone  # a number read in groups stays whole
talk = list(va.chunk_sentences(iter(["เรื่องนี้ต้องบอกตรงๆ ", "ครับว่าเลือกประเภทต้นทุนเป็น ", "LOT FIFO ได้ ", "แต่คิดจริงใช้ถัวเฉลี่ยครับ ", "ถามต่อได้ครับ"])))
assert talk[0].endswith("ถัวเฉลี่ยครับ "), talk  # one spoken sentence = one chunk, not cut at any space (2026-11-08 call)
real_judge, va.judge, turn = va.judge, (lambda sources, piece, before: False), {}  # a judge that cuts every claim
list(va.panel_events(iter(["สรุป\n| ขั้น | ทำอะไร |\n|---|---|\n| 1 | ไล่บรรทัด |\n"]), [], set(), "", turn))
va.judge = real_judge
assert turn["kept"][:2] == ["| ขั้น | ทำอะไร |", "|---|---|"], turn  # a table keeps its column names (2026-11-08 04:59)
# a table's own row numbers are no claim: rows 10, 11 and 13 of a 13-step table were cut (call 2026-11-09 06:38),
# 1-9 pass as lone digits. A first cell that does not count on from the row above is still a number to check
va.judge, turn = (lambda sources, piece, before: True), {}
rows = "| ขั้น | ทำอะไร |\n|---|---|\n" + "".join(f"| {i} | ไล่บรรทัด |\n" for i in range(1, 12)) + "| 75 | ไล่บรรทัด |\n"
list(va.panel_events(iter([rows]), [], set(), "", turn))
va.judge = real_judge
assert [r.split("|")[1].strip() for r in turn["kept"][2:] if r.startswith("|")] == [str(i) for i in range(1, 12)], turn["kept"]
# the STT spaces words out: "วันนี้ วัน อะไร" got no date (call 2026-11-09 06:39)
assert "วัน" in va.clock_note("วันนี้ วัน อะไร") and "เวลา" in va.clock_note("ตอนนี้ กี่ โมง") and not va.clock_note("ขอดูต้นทุน")
# an example's label line is no claim (ลุงจืด 2026-11-09), passed by code: in the judge's sources it cut small talk
va.judge, turn = (lambda sources, piece, before: False), {}
list(va.panel_events(iter([f"#### {C.SAMPLE_LABEL}\n### ตัวอย่างสมมติ (ไม่ใช่ข้อมูลจริง)\n| ขั้น | ทำอะไร |\n"]), [], set(), "", turn))
va.judge = real_judge
assert turn["kept"][:2] == [f"#### {C.SAMPLE_LABEL}", "### ตัวอย่างสมมติ (ไม่ใช่ข้อมูลจริง)"] and C.SAMPLE_LABEL not in va.OWN_WORDS, turn
va.judge, turn = (lambda sources, piece, before: "WH02" in piece), {}
list(va.panel_events(iter(["```\n- WH02 คลังสาขา\n```\n```sql\nSELECT x FROM bad_tbl\n```\n"]), [], set(), "", turn))
va.judge = real_judge
assert "- WH02 คลังสาขา" in turn["kept"] and "SELECT x FROM bad_tbl" not in turn["kept"], turn["kept"]
# a customer's message for the owner (2026-11-11): only a number the caller said is stored, the page escapes it all
C.LEADS_PATH, real_notify, va.notify_owner = os.path.join(tempfile.mkdtemp(), "leads.jsonl"), va.notify_owner, lambda: None
said = "ผมชื่อสมชัย เบอร์ ศูนย์แปดหนึ่ง 234 5678 ครับ"
assert va.spoken_digits(said) == "0812345678", va.spoken_digits(said)
assert "ยังไม่ได้บันทึก" in va.leave_message("สมชัย", phone="081-234-5678", heard="ผมชื่อสมชัยครับ")[1]  # a made-up number
assert "ยังไม่ได้บันทึก" in va.leave_message("", phone="081-234-5678", heard=said)[1]  # no name
assert "ยังไม่ได้บันทึก" in va.leave_message("สมชัย", phone="1234", heard="1234")[1] and not os.path.exists(C.LEADS_PATH)
card, told = va.leave_message("สมชัย", "<b>ร้าน</b>", "081-234-5678", "สนใจซื้อ\nโปรแกรม", heard=said, call="feed_c1")
assert told.startswith("บันทึกข้อความฝาก") and "081-234-5678" in told and card[0]["source"] == "message", told
lead = json.loads(open(C.LEADS_PATH, encoding="utf-8").read())
assert lead["phone"] == "081-234-5678" and lead["call"] == "feed_c1" and lead["message"] == "สนใจซื้อ โปรแกรม", lead
assert "&lt;b&gt;ร้าน&lt;/b&gt;" in va.leads_page() and "<b>" not in va.leads_page(), va.leads_page()
for _ in range(3):
    last = va.leave_message("สมชัย", phone="081-234-5678", heard=said, call="feed_c1")[1]
assert "ครบ 3 ครั้ง" in last and len(open(C.LEADS_PATH, encoding="utf-8").readlines()) == 3, last  # 3 per call
va.notify_owner = real_notify
# eval run 7: the judge's JSON was cut off (done_reason=length, its 4K context full); the retry reads the nearer
# half of the source, a second failure still cuts
from types import SimpleNamespace as NS
def fake_llm(replies):
    def chat(**kw):
        calls.append(kw["messages"][0]["content"])
        return NS(message=NS(content=replies.pop(0)), done_reason="length", prompt_eval_count=4090)
    return NS(chat=chat)
saved_llm, talk_json = getattr(va, "llm", None), '{"words": [{"word": "x", "type": "talk", "source_word": ""}]}'
src = "ต้นแหล่งข้อมูล " + "ข้อมูลทั่วไป " * 100 + "ท้ายแหล่งข้อมูล"
for replies, want in ((['{"words": [', talk_json], True), (['{"words": [', "oops"], False)):
    calls, va.llm = [], fake_llm(replies)
    assert va.judge(src, "ระบบเราติดตั้งให้ฟรีถึงร้านครับ", "") is want and len(calls) == 2, (want, len(calls))
    assert len(calls[1]) < len(calls[0]) and "ท้ายแหล่งข้อมูล" in calls[1] and "ต้นแหล่งข้อมูล" not in calls[1]
va.llm = saved_llm
sums = va.calc("รับเข้า 10 ชิ้น: 10*50\n1,200/20\n0.1+0.2\n1000/3\n1/0\n__import__('os')\n2**99999")[1].splitlines()
assert sums[:4] == ["รับเข้า 10 ชิ้น 10*50 = 500", "1200/20 = 60", "0.1+0.2 = 0.3", "1000/3 = 333.33"], sums  # Decimal, half up
assert all("คิดไม่ได้" in s for s in sums[4:7]) and len(sums) == 8, sums  # no crash, no code, no power; then the unit line
assert not va.grounding.unsourced("| 1 | 10 | 50.00 | 55.50 |", va.grounding.numbers("10*50 = 500 1100/20 = 55.5 50")), "50.00 is 50"
lot = va.calc("LOT A001 รับเข้า 100 ชิ้น ชิ้นละ 20 บาท: 100*20")[1]
assert not va.grounding.unsourced("| 1 | A001 | 100 | 20 | 2,000 |", va.grounding.numbers(lot)), lot  # a code in the label is a source
# call 09:15: every example row was cut over "001" of a made-up LOT A001; a code is a name, a rate or a spaced number is not
assert va.grounding.numbers("| 1 | รับเข้า | LOT A001 | 10 |") == {"10"}, va.grounding.numbers("LOT A001 10")
assert va.grounding.numbers("ภาษี VAT7%") == {"7#"} and va.grounding.numbers("LOT 001") == {"001"}
# call 09:50: calc gave 11.67, the spoken "สิบเอ็ดบาทหกสิบเจ็ดสตางค์" read 11 and 67, "สิบเอ็ดจุดหกเจ็ด" read 11.067
assert va.grounding.numbers("สิบเอ็ดบาทหกสิบเจ็ดสตางค์") == va.grounding.numbers("สิบเอ็ดจุดหกเจ็ด") == {"11.67"}
assert va.grounding.numbers("ห้าสิบบาทห้าสตางค์") == {"50.05"}
assert va.clean("ที่ต้องการ是关于อะไรครับ") == "ที่ต้องการอะไรครับ"  # DeepSeek slipped Chinese into a Thai sentence (log 06:34)
src = "ic_inventory.balance_qty qty_2 calc_flag flag: 12, 310, 54-73"
assert va.code_unsourced("SELECT SUM(COALESCE(d.qty_2,0) * d.calc_flag) AS stock_value FROM ic_inventory i",
                         va.grounding.numbers(src), src) == set()  # its own AS name, and qty_2,0 is no number 20
assert va.code_unsourced("AND d.calc_flag IN (12, 310, 54, 55, 72, 73)", va.grounding.numbers(src), src) == set()
assert va.code_unsourced("FROM ic_stock_balance s WHERE x = 999", va.grounding.numbers(src), src) == {"ic_stock_balance", "999"}
sql_own = set()
assert va.code_unsourced("WITH stock AS (", set(), src, sql_own) == set() and "stock" in sql_own
assert va.code_unsourced("SELECT SUM(d.qty_2)::numeric(18,2) AS qty_in FROM ic_inventory d)", set(), src, sql_own) == set()
assert va.code_unsourced("SELECT qty_in FROM stock", set(), src, sql_own) == set()  # numeric(18,2) + earlier CTE/AS pass
assert len(va.calc("\n".join(f"แถว {i}: {i}*10" for i in range(1, 36)))[1].splitlines()) == 36  # supports 30+ row tables
chat, va.worker_chat = va.worker_chat, lambda messages, tools: ("LOT อัตโนมัติ\nเปลี่ยนเรื่อง: LOT FIFO | ระบบรถยนต์", [])
talk = [{"role": "user", "content": "LOT FIFO คิดยังไง"}, {"role": "assistant", "content": "คิดแบบถัวเฉลี่ยครับ"}]
assert va.search_words(talk, "รถอัตโนมัติ") == ("LOT อัตโนมัติ", ("LOT FIFO", "ระบบรถยนต์"))
assert va.search_words([], "รถอัตโนมัติ")[1] is None  # a first question has no topic to leave
va.worker_chat = lambda messages, tools: ("FIFO ตาราง", [])
assert va.search_words(talk, "ตารางไฟโฟ") == ("FIFO ตาราง", None)
# log 2026-11-09: the jump line first, or a reply sentence for a query; neither is searched
va.worker_chat = lambda messages, tools: ("เปลี่ยนเรื่อง: LOT FIFO | ระบบรถยนต์\nLOT อัตโนมัติ", [])
assert va.search_words(talk, "รถอัตโนมัติ") == ("LOT อัตโนมัติ", ("LOT FIFO", "ระบบรถยนต์"))
va.worker_chat = lambda messages, tools: ("ได้เลยครับ เดี๋ยวผมค้นให้นะครับ.", [])
assert va.search_words(talk, "ตารางไฟโฟ") == ("", None)
import numpy as np
def no_boss(*a, **k):
    raise AssertionError("the topic ask called the boss")
saved = va.early_words, va.speak, va.worker_chat, va.llm_tokens, va.SESSIONS.get("")
va.early_words, va.speak, va.llm_tokens = (lambda data: "รถอัตโนมัติ"), (lambda text: np.zeros(1)), no_boss
va.worker_chat = lambda messages, tools: ("LOT อัตโนมัติ\nเปลี่ยนเรื่อง: LOT FIFO | ระบบรถยนต์", [])
va.SESSIONS[""] = list(talk)
bots = [o[1].args[0]["text"] if isinstance(o, tuple) else None for o in va.reply((16000, np.zeros(16000)))]
assert [b for b in bots if b] == ["ขอถามให้ชัดก่อนครับ จะเปลี่ยนไปคุยเรื่องระบบรถยนต์ หรือยังคุยเรื่องLOT FIFOต่อครับ"], bots
assert va.SESSIONS[""][-1]["content"].startswith("ขอถามให้ชัดก่อน")  # the ask is in the history: the next turn knows it
seen = []
def boss(messages, calls, tools, choice=None):
    seen.append(messages[-1]["content"])
    return iter(())
def no_search(calls, said, turn, tools):
    return [""] * len(calls)
    yield
dispatch, va.dispatch, va.llm_tokens, va.early_words = va.dispatch, no_search, boss, (lambda data: "ไม่ใช่ครับ ยังเรื่องเดิม")
bots = [o[1].args[0]["text"] if isinstance(o, tuple) else None for o in va.reply((16000, np.zeros(16000)))]
assert "ขอถามให้ชัดก่อน" not in " ".join(b for b in bots if b), bots  # never the topic ask twice in a row
assert "คือคุยเรื่องLOT FIFOต่อ" in seen[0], seen  # eval r18: "ยังเรื่องเดิม" got the new topic back
# call 2026-11-08 13:16: 28 of 92 lines were a bare "ขอโทษครับ" opener, all a talked-over caller heard. It is dropped,
# the rest is still said, and an opener alone falls to CUT_ALL, never silence
va.judge, va.early_words = (lambda *a: True), (lambda data: "ฟังไม่ออก")
for toks, want in ((["ขอโทษ", "ครับ ", "ฟังไม่ชัด", "เลย ช่วยพูดอีกครั้ง", "ได้ไหมครับ"], ["ฟังไม่ชัดเลย ช่วยพูดอีกครั้งได้ไหมครับ"]),
                   (["ขอโทษ", "ครับ"], [va.CUT_ALL])):
    va.SESSIONS[""], va.llm_tokens = [], (lambda *a, toks=toks: iter(toks))
    bots = [o[1].args[0]["text"] for o in va.reply((16000, np.zeros(16000))) if isinstance(o, tuple)]
    assert bots == want, bots
# call 2026-11-14 05:44..06:08: a lone "ได้ครับ " joins the next sentence; if the sentence is cut, a valid [จอ] gets
# SCREEN_ONLY (never a clipped "ได้ครับ" or false yes, never "หาไม่เจอ"), and no [จอ] falls to CUT_ALL
va.SESSIONS[""], va.llm_tokens = [], (lambda *a: iter(["ได้ครับ ", "ฟังเสียงของคุณชัดเจนดีมากเลย", "ครับ"]))
assert [o[1].args[0]["text"] for o in va.reply((16000, np.zeros(16000))) if isinstance(o, tuple)] == ["ได้ครับ ฟังเสียงของคุณชัดเจนดีมากเลยครับ"]
va.judge = lambda sources, piece, before: False
va.SESSIONS[""], va.llm_tokens = [], (lambda *a: iter(["ได้ครับ ", "ประโยคแต่งที่ยาวเกินยี่สิบตัว", "ครับ ", "[จอ]\n| ลำดับ | รายการ |\n|---|---|\n"]))
assert [o[1].args[0]["text"] for o in va.reply((16000, np.zeros(16000))) if isinstance(o, tuple)] == [va.SCREEN_ONLY]
va.SESSIONS[""], va.llm_tokens = [], (lambda *a: iter(["ได้ครับ ", "ประโยคแต่งที่ยาวเกินยี่สิบตัว", "ครับ"]))
assert [o[1].args[0]["text"] for o in va.reply((16000, np.zeros(16000))) if isinstance(o, tuple)] == [va.CUT_ALL]
va.judge = real_judge
va.dispatch = dispatch
va.early_words, va.speak, va.worker_chat, va.llm_tokens, va.SESSIONS[""] = saved
va.worker_chat = chat
# AirPods call 2026-11-08 20:32: a lone เอ่อ/นะครับ was a turn that cut the answer off; อืม passes over only while it talks or thinks
from concurrent.futures import Future
from types import SimpleNamespace as NS
h = NS(playing=(), HESITATIONS=va.BargeInReplyOnPause.HESITATIONS, BACKCHANNELS=va.BargeInReplyOnPause.BACKCHANNELS)
for heard, responding, playing, want in (("เอ่อ...", False, (), True), ("นะครับ", False, (), True), ("อืม", False, (), False),
                                          ("อืม", True, (), True), ("อืม", False, ((0, time.monotonic(), "x"),), True),
                                          ("หืม", True, (), False), ("ขอดูต้นทุน", True, (), False), (None, False, (), False),
                                          ("เสียงดนตรี", False, (), True)):  # Whisper's music caption, 22 turns in the log
    va.EARLY.clear()
    if heard is not None:
        (f := Future()).set_result(heard)
        va.EARLY[id(h)] = (None, f)
    h.playing = playing
    state = NS(stream=np.zeros(16000), sampling_rate=16000, responding=responding)
    assert va.BargeInReplyOnPause.just_filler(h, state) is want, (heard, responding, playing)
va.EARLY.clear()
# AirPods call 2026-11-08 23:03: the turn checks ran on every frame until the reply started (3-8 times a turn).
# They run once a pause
from fastrtc.reply_on_pause import AppState
h, checks = object.__new__(va.BargeInReplyOnPause), []
h.determine_pause, h.send_message_sync = (lambda *a: True), (lambda m: None)
h.hears_itself = lambda state: False
h.just_filler = lambda state: checks.append(1)  # None: a turn
h.state = AppState(stream=np.zeros(16000, np.int16), sampling_rate=16000)
voiced, va.voiced = va.voiced, (lambda audio: np.ones(8000, np.float32))
for _ in range(3):
    va.BargeInReplyOnPause.process_audio(h, (16000, np.zeros(320, np.int16)), h.state)
assert h.state.pause_detected and len(checks) == 1, checks
def stt_down(state):
    raise RuntimeError("CUDA error")
h.just_filler, h.state = stt_down, AppState(stream=np.zeros(16000, np.int16), sampling_rate=16000)
va.BargeInReplyOnPause.process_audio(h, (16000, np.zeros(320, np.int16)), h.state)
assert h.state.pause_detected  # an STT error in the check is a turn, not the end of listening
pushed, va.PENDING[""]["text"] = [], "ขอดูต้นทุน"  # typed while speaking, and the speech was noise: the typed turn starts
h.just_filler, h.push_turn, h.state = (lambda state: True), (lambda interrupt: pushed.append(interrupt)), AppState(stream=np.zeros(16000, np.int16), sampling_rate=16000)
va.BargeInReplyOnPause.process_audio(h, (16000, np.zeros(320, np.int16)), h.state)
assert not h.state.pause_detected and pushed == [False], pushed
va.PENDING[""]["text"] = ""
va.voiced = voiced
# nothing heard (or only Whisper's music caption): the page still gets "done", never stuck on กำลังพูด…
saved = va.early_words, va.transcribe
for early in ("", None):
    va.early_words, va.transcribe = (lambda data, early=early: early), (lambda audio: "เสียงดนตรี")
    outs = [o.args[0] for o in va.reply((16000, np.zeros(16000)))]
    assert outs == [{"kind": "done"}], outs
va.early_words, va.transcribe = saved
# bcbrain down at startup: no prefetch tools, no ThreadPoolExecutor(0); a barge-in does not wait for the workers
gen = va.dispatch([], [], {}, {})
try:
    next(gen)
except StopIteration as e:
    assert e.value == []
gen, t = va.dispatch([("slow", {})], ["x"], {}, {"slow": lambda: time.sleep(1.5)}), time.perf_counter()
next(gen)
gen.close()
assert time.perf_counter() - t < 1, time.perf_counter() - t
assert list(va.FILLER_TEXT) == ["hello"]
outs = list(va.dispatch([("web_search", {"query": "x"})], [], {}, {"web_search": lambda query: ([], "ok")}))
assert not any(isinstance(o, tuple) for o in outs), outs  # no spoken "ขอตรวจสอบ/ขอค้น…" filler before the answer
# log 2026-11-09: a plain sum went without calc; a phone's dashes are no sum
assert all(va.SUM_ASK.search(va.masked(s)) for s in ("50 หาร 0.5 บวก 50", "120 x 3 เท่าไหร่", "ลบ 7 แล้วเหลือเท่าไหร่"))
assert not any(va.SUM_ASK.search(va.masked(s)) for s in ("เบอร์ 081-123-4567 ครับ", "ต้นทุนแบบถัวเฉลี่ย", "ลบเอกสารยังไง"))
assert va.swarm_queries("ลูกหนี้ หักกลบ ; รายงานลูกหนี้ | ลูกหนี้ หักกลบ", "x", []) == ["ลูกหนี้ หักกลบ", "รายงานลูกหนี้"]
assert va.CALC_ASK.search("วิธีคำนวณต้นทุนสินค้า") and not va.CALC_ASK.search("เก็บได้กี่คลังสินค้า กี่ location")
assert va.CALC_ASK.search("วิธีคิดภาษีมูลค่าเพิ่ม") and not va.CALC_ASK.search("ออกใบกำกับภาษีขายพิมพ์ผิดยกเลิกยังไง")
assert va.FAST_CHAT.match("สวัสดีครับ") and not va.FAST_CHAT.match("สวัสดีครับ ขายวัสดุก่อสร้าง")
assert not va.grounding.new_words("[การ์ด] ลูกหนี้: 50,000 | เจ้าหนี้: 30,000", f"{va.OWN_WORDS} ลูกหนี้ เจ้าหนี้")
# TALK_WORDS: the judge still sees every made-up line, and asks to repeat/spell/explain need no judge
from eval_talk import MUST_CUT
assert all(va.grounding.new_words(line, f"{va.OWN_WORDS}\n{va.TALK_WORDS}") for _, line in MUST_CUT)
for line, term in (("ช่วยสะกดหรือบอกอีกนิดได้ไหมครับ", ""), ("ผมไม่แน่ใจว่าหมายถึงจาวาสคริปต์อะไรครับ", "จาวาสคริปต์"),
                   ("ผมยังไม่เข้าใจคำว่าแฟลตกับออนเคานต์นี้ ช่วยอธิบายสั้นๆ ได้ไหมครับ", "แฟลต ออนเคานต์")):
    assert not va.grounding.new_words(line, f"{va.OWN_WORDS}\n{va.TALK_WORDS}\n{term}"), line
# the who-line is no judge source (A/B 2026-11-10: it flipped small talk 3/5 -> 2/5); the line before it is
facts = va.setup_facts(va.BASE_PROMPT + "\nสายนี้ค้นเว็บไม่ได้" + va.ABOUT_ASK + "ผมเป็นเลขา AI")
assert "สายนี้ค้นเว็บไม่ได้" in facts and "เลขา AI" not in facts, facts
assert va.LONG_NUMBER.sub("", "ร้านสมหญิงวัสดุ เบอร์ 081-123-4567 ค่ะ ปี 2567") == "ร้านสมหญิงวัสดุ เบอร์  ค่ะ ปี 2567"  # no phone to the BC brain
assert va.LONG_NUMBER.sub("", "เบอร์ ศูนย์ แปด หนึ่ง หนึ่ง สอง สาม สี่ ห้า หก เจ็ด ค่ะ ภาษีสองแบบ") == "เบอร์ ค่ะ ภาษีสองแบบ"
assert va.clean("**สวัสดี** # ครับ\n- ข้อหนึ่ง") == "สวัสดี ครับ ข้อหนึ่ง"
screen = []
toks = iter(["สรุปว่า", "พร้อมครับ ", "[จ", "อ]\n# หัวข้อ", "\n- ก", "\n- ข"])
assert "".join(va.until_screen(toks, screen)) == "สรุปว่าพร้อมครับ ", screen  # marker split across tokens
assert screen == ["\n# หัวข้อ"] and list(toks) == ["\n- ก", "\n- ข"]  # rest stays in the stream for the panel
assert "".join(va.until_screen(iter(["ไม่มี [", "ข้อมูลจอ"]), [])) == "ไม่มี [ข้อมูลจอ"  # partial marker released

C.DOCS_DIR = tempfile.mkdtemp()
C.SESSIONS_DIR = tempfile.mkdtemp()  # the call's session file goes here, not into sessions/
with open(os.path.join(C.DOCS_DIR, "ร้าน.txt"), "w", encoding="utf-8") as f:
    f.write("ข้อมูลทั่วไป\n" * 200 + "ร้านกาแฟลุงจืด เปิดทุกวัน เวลาเจ็ดโมงเช้าถึงหกโมงเย็น ปิดทุกวันพุธ\n")
with open(os.path.join(C.DOCS_DIR, "broken.pdf"), "wb") as f:
    f.write(b"not a pdf")
cards, _ = va.search_docs("ร้านกาแฟ เวลาเปิด")
assert cards and "เจ็ดโมง" in cards[0]["snippet"], cards
pub_app = va.make_app(va.make_stream(True), public=True)
pub_routes = {getattr(r, "path", None) for r in pub_app.routes}
assert {"/robots.txt", "/sitemap.xml", "/llms.txt", "/ice"} <= pub_routes and "/leads" not in pub_routes, pub_routes
assert "Googlebot" in va.ROBOTS_TXT and "OAI-SearchBot" in va.ROBOTS_TXT and "https://javis.bcsmethai.com/sitemap.xml" in va.ROBOTS_TXT
assert "<loc>https://javis.bcsmethai.com/</loc>" in va.SITEMAP_XML and "BC AI Account" in va.LLMS_TXT

# --- models + server ---------------------------------------------------------------------------------------------
va.stt = va.load_stt()
va.llm = ollama.Client(host=C.OLLAMA_HOST)
va.tts = va.load_tts()
va.load_mcp_tools()
va.load_workers()
# the public line (javis): BC brain, calc, general knowledge and leave_message only; this PC's own tools stay out
pub = va.PUBLIC["tools"]
assert {"calc", "general_knowledge", "leave_message"} <= set(pub) and {va.WORKER[n] for n in pub} <= {"bcbrain", "calc", "deepseek", "message"}, sorted(pub)
assert {s["function"]["name"] for s in va.PUBLIC["specs"]} == set(pub)
assert all("ไม่สำเร็จ" in va.run_tool(n, {"query": "x"}, pub)[1] for n in ("search_docs", "web_search", "desktop_task"))
assert "desktop_task" not in va.PUBLIC["prompt"] and "Gmail" not in va.PUBLIC["prompt"]
assert va.NOTE_META.sub("", "# ก\nslug: a | folder: x\nsources: f.ts:1\n\n# ก\nเนื้อ") == "# ก\n\n# ก\nเนื้อ"  # customers: no code paths
assert va.MAP_LINE.match(" [แผนที่] เยาวราช กรุงเทพมหานคร ")[1] == "เยาวราช กรุงเทพมหานคร"  # the line the server adds "@lat,lon" to
if C.TURN_SECRET:  # coturn use-auth-secret: "<expiry>:javis" + base64 HMAC-SHA1, valid about an hour
    turn = va.ice_servers()["iceServers"][1]
    assert 3500 < int(turn["username"].split(":")[0]) - time.time() <= 3600 and len(base64.b64decode(turn["credential"])) == 20
# barge-in gate: the caller's speech counts, 1.2 s of hiss over the bot does not (2026-11-04 a noise blip cut a whole answer)
speech = (librosa.load(C.REF_AUDIO_PATH, sr=48000)[0][: 2 * 48000] * 32767).astype(np.int16)
hiss = np.random.default_rng(0).normal(0, 800, 57600).astype(np.int16)
assert va.stt_hears((48000, speech)) and not va.stt_hears((48000, hiss)) and not va.stt_hears((48000, hiss * 0))
va.warm_up()
# endpointing: the turn ends PAUSE_SILENCE_S (+ one check step) after the last word, its words transcribed already
h, early = va.make_stream().event_handler, set()  # early: sizes of the audio each early transcription started on


def until_pause(*parts):
    h.state = h.state.new()
    early.clear()
    said = np.concatenate([hiss * 0, *parts, np.zeros(48000, np.int16)])  # 1.2 s silence first, 1 s at the end
    for n in range(0, said.size, 960):  # 20 ms frames, as WebRTC sends them
        h.process_audio((48000, said[n : n + 960]), h.state)
        early.update([va.EARLY[id(h)][0].size] if id(h) in va.EARLY else [])
        if h.state.pause_detected:
            return (n + 960 - said.size) / 48000 + 1  # seconds after the last word
    raise AssertionError("no pause")


pause_after = until_pause(speech)
print(f"endpointing: pause {pause_after:.2f}s after the speech, turn audio {h.state.stream.size / 48000:.1f}s")
assert pause_after <= C.PAUSE_SILENCE_S + 0.2, pause_after
assert h.state.stream.size <= 4.2 * 48000, "the turn starts at most a second before the first word"
assert id(h) in va.EARLY, "no early transcription started before the turn ended"
# the reply is busy and nobody takes the turn yet: quiet must not re-run the STT (it did every 0.9 s on the whole,
# growing turn: 36 s of audio once a second, 20 s before a caller's words came out)
head = va.EARLY[id(h)][0].size
for _ in range(150):  # 3 s
    h.process_audio((48000, np.zeros(960, np.int16)), h.state)
assert va.EARLY[id(h)][0].size == head, "quiet after the turn re-ran the STT"
assert va.early_words(h.state.stream) == va.transcribe((48000, h.state.stream)) != "" and not va.EARLY
until_pause(speech, np.zeros(int(0.3 * 48000), np.int16), speech)  # a pause mid-turn starts one, then more words
assert len(early) == 2, (early, h.state.stream.size / 48000)
assert va.early_words(h.state.stream) == va.transcribe((48000, h.state.stream)), "early words missed the speech after them"
# no headphones: the secretary's own voice back through the mic while it plays is no turn; other words still are
h.playing = ((time.monotonic() - 9, time.monotonic() + 60, va.transcribe((48000, speech))),)
try:
    until_pause(speech)
    raise SystemExit("its own voice made a turn")
except AssertionError:
    pass
h.playing = ((time.monotonic() - 9, time.monotonic() + 60, "ราคาทองวันนี้เท่าไหร่ครับ"),)
until_pause(speech)
h.playing = ()
# words for someone else in the room: the boss answers only SILENT_MARK (live DeepSeek), and that makes no sound
room = [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": "คุณต่าย คุณต่าย คุณต่าย"}]
assert C.SILENT_MARK in "".join(va.llm_tokens(room, [], None)), "the boss answered someone else in the room"
real, va.llm_tokens = va.llm_tokens, lambda *a: iter([C.SILENT_MARK])
spoken = [o for o in va.reply((48000, speech)) if isinstance(o, tuple) and isinstance(o[0], tuple)]
va.llm_tokens = real
assert not spoken and va.SESSIONS[""][-1]["content"] == C.SILENT_MARK, spoken
assert all(va.speak("ครับ").size for _ in range(20)), "a short word came out as 0 samples"  # it did 3 of 20 times
# DeepSeek sometimes writes a tool call as its own DSML text (a call spoke "<｜｜DSML｜｜ invoke ..." and searched
# nothing): none of it reaches the caller, and the call still runs
DSML_TOKENS = ["เข้าใจครับ <", '｜｜DSML｜｜calls>\n<｜｜DSML｜｜invoke name="read_note">\n<｜｜DSML｜｜parameter name="slug" '
               'string="true">pos-overview</｜｜DSML｜｜parameter>\n<｜｜DSML｜｜parameter name="limit" string="false">5',
               "</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜calls>"]


class FakeDeepSeek:  # the SSE stream of that reply
    def stream(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def raise_for_status(self): pass
    def iter_lines(self): return (f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in DSML_TOKENS)


real, va.DEEPSEEK, calls = va.DEEPSEEK, FakeDeepSeek(), []
said = "".join(va.deepseek_tokens([], calls, [{}]))
va.DEEPSEEK = real
assert said.strip() == "เข้าใจครับ" and [(c["name"], c["args"]) for c in calls] == [("read_note", {"slug": "pos-overview", "limit": 5})], (said, calls)
# replay 2026-11-08: the last-round markup had a space after the bars, "<｜｜DSML｜｜ invoke", and read as no call
DSML_TOKENS = [t.replace("｜｜DSML｜｜", "｜｜DSML｜｜ ") for t in DSML_TOKENS]
va.DEEPSEEK, calls = FakeDeepSeek(), []
said = "".join(va.deepseek_tokens([], calls, [{}]))
va.DEEPSEEK = real
assert [(c["name"], c["args"]) for c in calls] == [("read_note", {"slug": "pos-overview", "limit": 5})], (said, calls)
# the last round sees the turn as plain text: with its tool calls kept DeepSeek wrote another as DSML (replay 2026-11-08)
plain = va.plain_turn([{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                       {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                       {"role": "tool", "tool_call_id": "1", "content": "r1"},
                       {"role": "assistant", "content": "ขอเช็กให้ครับ", "tool_calls": [{"id": "2"}]},
                       {"role": "tool", "tool_call_id": "2", "content": "r2"}, va.LAST_ROUND])
assert [m["role"] for m in plain] == ["system", "user", "assistant", "user"] and "r1" in plain[1]["content"], plain
assert not any("tool_calls" in m for m in plain) and plain[3]["content"].endswith(va.LAST_ROUND["content"]), plain
# the judge's second look reads the source lines nearest the piece, a one-line note's \n breaks too
near = va.grounding.near_sources("ยกเลิกบิลขาย", "หัวข้อ\\nวิธียกเลิกบิลขายที่หน้าจอขาย\\nอื่นๆ\n" + "ก" * 9000, 100)
assert near == "วิธียกเลิกบิลขายที่หน้าจอขาย", near
# DeepSeek down (here: a bad key) -> the caller hears the apology, nothing made up
key, C.DEEPSEEK_API_KEY = C.DEEPSEEK_API_KEY, "sk-invalid"
assert "".join(va.llm_tokens([{"role": "user", "content": "พูดคำว่า สวัสดี"}], [], None)) == va.BRAIN_DOWN
C.DEEPSEEK_API_KEY = key
threading.Thread(target=uvicorn.run, args=(va.make_app(va.make_stream()),),
                 kwargs=dict(host="127.0.0.1", port=7861, log_level="warning"), daemon=True).start()
time.sleep(2)
BASE = "http://127.0.0.1:7861"


async def tts_clip(text):
    mp3 = os.path.join(C.DOCS_DIR, "..", f"{uuid.uuid4().hex}.mp3")
    for wait in (2, 5, 10, 20, 0):  # the caller's voice service answers 503 now and then
        try:
            await edge_tts.Communicate(text, C.REF_VOICE).save(mp3)
            break
        except Exception:
            if not wait:
                raise
            await asyncio.sleep(wait)
    clip = (librosa.load(mp3, sr=48000)[0] * 32767).astype(np.int16)
    os.remove(mp3)
    return clip


class Mic(MediaStreamTrack):  # 20 ms frames at 48 kHz in real time; plays queued clips, else silence
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.n, self.t0, self.clip, self.pos = 0, None, None, 0

    def say(self, clip):
        self.clip, self.pos = clip, 0

    async def recv(self):
        self.t0 = self.t0 or time.time()
        await asyncio.sleep(max(0, self.t0 + self.n * 0.02 - time.time()))
        out = np.zeros(960, np.int16)
        if self.clip is not None:
            seg = self.clip[self.pos: self.pos + 960]
            out[: len(seg)] = seg
            self.pos += 960
            if self.pos >= len(self.clip):
                self.clip = None
        f = AudioFrame.from_ndarray(out.reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts, f.time_base = 48000, self.n * 960, fractions.Fraction(1, 48000)
        self.n += 1
        return f


async def main():
    q_docs = await tts_clip("ช่วยดูในเอกสารหน่อยครับ ว่าร้านกาแฟลุงจืดเปิดกี่โมง")
    q_sum = await tts_clip("ช่วยคิดหน่อยครับ สินค้าชิ้นละห้าสิบบาท ซื้อสามชิ้น เป็นเงินเท่าไหร่")
    barge = (librosa.load(C.REF_AUDIO_PATH, sr=48000)[0][: int(1.3 * 48000)] * 32767).astype(np.int16)
    mic, pc, got, events = Mic(), RTCPeerConnection(), [], []
    pc.addTrack(mic)
    pc.createDataChannel("text")
    wid = uuid.uuid4().hex[:10]

    @pc.on("track")
    def on_track(track):
        async def pull():
            while True:
                try:
                    fr = await track.recv()
                except MediaStreamError:  # call hung up
                    return
                a = fr.to_ndarray().astype(np.float32) / 32768
                got.append((time.time(), a.reshape(-1, len(fr.layout.channels))[:, 0], fr.sample_rate))
        asyncio.ensure_future(pull())

    async def listen():
        async with httpx.AsyncClient(timeout=None) as c, c.stream("GET", f"{BASE}/outputs", params={"webrtc_id": wid}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    events.append((time.time(), json.loads(line[5:])))

    async def wait_for(pred, timeout=90):
        start = len(events)
        for _ in range(timeout * 10):
            if any(pred(e) for _, e in events[start:]):
                return
            await asyncio.sleep(0.1)
        raise AssertionError(f"timed out; events so far: {[e for _, e in events[start:]]}")

    await pc.setLocalDescription(await pc.createOffer())
    body = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "webrtc_id": wid}
    ans = httpx.post(f"{BASE}/webrtc/offer", json=body, timeout=30).json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    asyncio.ensure_future(listen())
    await asyncio.sleep(1)

    await wait_for(lambda e: e["kind"] == "done")  # greeting: the secretary speaks first
    assert next(e for _, e in events if e["kind"] == "user")["text"] == "", "greeting must not show a caller line"
    greet_end = len(events)

    turns = []  # (spoken_at_end, first_bot_event_time)
    mic.say(q_docs)
    said_end = time.time() + len(q_docs) / 48000
    await wait_for(lambda e: e["kind"] == "done")
    turns.append(("docs", said_end))

    mic.say(q_sum)
    said_end = time.time() + len(q_sum) / 48000
    turns.append(("sum", said_end))
    await wait_for(lambda e: e["kind"] == "bot")
    mic.say(barge)  # talk over the bot
    await wait_for(lambda e: e["kind"] == "user")
    await wait_for(lambda e: e["kind"] == "done")


    assert httpx.post(f"{BASE}/chat", json={"webrtc_id": wid, "text": "  "}).status_code == 400
    assert httpx.post(f"{BASE}/chat", json={"webrtc_id": "nobody", "text": "สวัสดี"}).status_code == 409
    assert httpx.post(f"{BASE}/chat", json={"webrtc_id": wid, "text": "ใบกำกับภาษีขายยกเลิกยังไง"}).status_code == 200
    await wait_for(lambda e: e["kind"] == "user" and "ใบกำกับภาษี" in e["text"])
    await wait_for(lambda e: e["kind"] == "done")
    team = httpx.get(f"{BASE}/team").json()
    await asyncio.sleep(2)
    await pc.close()

    sf.write("out_call.wav", np.concatenate([a for _, a, _ in got]), got[0][2])
    kinds = [e["kind"] for _, e in events]
    for name, end in turns:
        first = next(t for t, e in events if e["kind"] == "bot" and t > end)
        print(f"{name}: end of question -> first bot phrase {first - end:.2f}s")
    print("event kinds:", {k: kinds.count(k) for k in set(kinds)})
    for _, e in events:
        if e["kind"] in ("user", "bot", "status", "info"):
            print(" ", e["kind"], (e.get("text") or f"[{e.get('source')}] {e.get('title')}: {str(e.get('snippet'))[:80]}")[:120])
    panels = [e["text"] for _, e in events if e["kind"] == "panel"]
    print("last panel:", panels[-1][:1500] if panels else None)

    info = [e for _, e in events if e["kind"] == "info"]
    assert any(c["source"] == "docs" for c in info), "search_docs was not used"
    turn1 = events[greet_end:]
    first_answer = " ".join(e.get("text", "") for _, e in turn1[: [e["kind"] for _, e in turn1].index("done")] if e["kind"] in ("bot", "panel"))
    assert "เจ็ด" in first_answer or "7" in first_answer, first_answer
    jobs = [e for _, e in events if e["kind"] == "team"]
    runs = {e["id"]: e["worker"] for e in jobs if e["state"] == "run"}
    assert "docs" in runs.values() and all(any(e["id"] == i and e["state"] != "run" for e in jobs) for i in runs), jobs
    assert "boss" in team and {"web", "docs"} <= set(team["workers"]) and "camera" not in team["workers"], team
    if "search_brain" in va.TOOLS:  # BC question: the knowledge base is searched before the brain speaks
        assert "bcbrain" in runs.values(), jobs
        brain = [c for c in info if c["source"] == "bcbrain"]  # every card names its worker; BC notes open in full
        assert all(c.get("job") in runs for c in info) and any(c.get("slug") for c in brain), brain
        note = httpx.get(f"{BASE}/note", params={"slug": next(c["slug"] for c in brain if c.get("slug"))})
        assert note.status_code == 200 and len(note.json()["text"]) > 500, note.text[:200]
        assert httpx.get(f"{BASE}/note", params={"slug": "../x"}).status_code == 404
    print("team:", team, "jobs:", [(runs[e["id"]], e["state"], e.get("secs")) for e in jobs if e["state"] != "run"])
    saved = json.load(open(os.path.join(C.SESSIONS_DIR, f"{wid}.json"), encoding="utf-8"))
    assert saved == va.SESSIONS[wid], "session file differs from the live history"
    users = [m["content"] for m in saved if m["role"] == "user"]
    assert len(users) == 5 and users[0].startswith("(ระบบ:"), f"expected greeting, docs, sum, barge-in, chat; got {users}"
    print("OK turns:", users)


asyncio.run(main())
