"""End-to-end check without a mic or camera: a real WebRTC call (aiortc client) into the same app voice_agent serves.

Turn 0 is the secretary's own greeting when the call connects, turn 1 asks about a local document (search_docs),
turn 2 asks what the camera sees (a red circle frame) and is talked over (barge-in), turn 3 is a two-finger choice
gesture with no speech, turn 4 is typed chat about BC (knowledge base first). Subtitles/cards/panel/team come over SSE.
Bot audio is saved to out_call.wav. Stop voice_agent.py first (this loads its own models).
Run: python test_call.py
"""
import asyncio
import base64
import fractions
import io
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
from PIL import Image, ImageDraw

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
cards, _ = va.search_docs("ร้านกาแฟ เวลาเปิด")
assert cards and "เจ็ดโมง" in cards[0]["snippet"], cards
ui = open(os.path.join(C.BASE_DIR, "static", "index.html"), encoding="utf-8").read()
assert all(f'"{g}"' in ui or g.startswith("fingers_") for g in va.GESTURES), "gesture ids out of sync with the UI"

# --- models + server ---------------------------------------------------------------------------------------------
va.stt = va.load_stt()
va.llm = ollama.Client(host=C.OLLAMA_HOST)
va.tts = va.load_tts()
va.voice = va.load_voice()
va.load_mcp_tools()
va.load_workers()
# the public line (javis): the BC brain only, in code; this PC's own tools answer as unknown there
pub = va.PUBLIC["tools"]
assert pub and all(va.WORKER[n] == "bcbrain" for n in pub), sorted(pub)
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
# several people near the mic (ลุงจืด 2026-11-04 "เสียงมีหลายคนพูด"): the call's first voice is the caller's, another
# voice later is no turn; the same voice saying other words and a short ครับ still are
h.caller = None
until_pause(speech)
other = (librosa.effects.pitch_shift(speech / 32767, sr=48000, n_steps=7) * 32767).astype(np.int16)  # 0.12 to the caller
try:
    until_pause(other)
    raise SystemExit("another voice made a turn")
except AssertionError:
    pass
until_pause((librosa.load(C.REF_AUDIO_PATH, sr=48000)[0][2 * 48000 :] * 32767).astype(np.int16))  # 0.59: the same voice
until_pause(other[int(0.5 * 48000) : int(0.85 * 48000)])  # 0.29 s of speech, too short to tell: a turn
h.caller = None
# words for someone else in the room: the boss answers only SILENT_MARK (live DeepSeek), and that makes no sound
room = [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": "คุณต่าย คุณต่าย คุณต่าย"}]
assert C.SILENT_MARK in "".join(va.llm_tokens(room, [], None)), "the boss answered someone else in the room"
real, va.llm_tokens = va.llm_tokens, lambda *a: iter([C.SILENT_MARK])
spoken = [o for o in va.reply((48000, speech)) if isinstance(o, tuple) and isinstance(o[0], tuple)]
va.llm_tokens = real
assert not spoken and va.SESSIONS[""][-1]["content"] == C.SILENT_MARK, spoken
assert all(va.speak("ครับ").size for _ in range(20)), "a short word came out as 0 samples"  # it did 3 of 20 times
# the camera filler said ภาพ first, so the answer may say it too (the judge cut a true camera answer over it)
assert not va.grounding.new_words("จากภาพตอนนี้เห็นเป็นสีขาวกับวงกลมสีแดงครับ", "ตอนนี้ในกล้องเห็นอะไรบ้างครับ\nสีขาว วงกลมสีแดง\n" + va.OWN_WORDS)
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


img = Image.new("RGB", (640, 480), "white")
ImageDraw.Draw(img).ellipse((200, 120, 440, 360), fill="red")
buf = io.BytesIO()
img.save(buf, "JPEG", quality=70)
RED_JPG = buf.getvalue()


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
    q_cam = await tts_clip("ตอนนี้ในกล้องเห็นอะไรบ้างครับ")
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

    httpx.post(f"{BASE}/frame", content=RED_JPG, headers={"Content-Type": "image/jpeg"})
    mic.say(q_cam)
    said_end = time.time() + len(q_cam) / 48000
    await wait_for(lambda e: e["kind"] == "info" and e["source"] == "camera")
    turns.append(("camera", said_end))
    await wait_for(lambda e: e["kind"] == "bot")
    mic.say(barge)  # talk over the bot
    await wait_for(lambda e: e["kind"] == "user")
    await wait_for(lambda e: e["kind"] == "done")

    for bad in ("nonsense", "thumbs_up"):  # only finger choices exist now
        assert httpx.post(f"{BASE}/gesture", json={"webrtc_id": wid, "gesture": bad}).status_code == 400
    httpx.post(f"{BASE}/gesture", json={"webrtc_id": wid, "gesture": "fingers_2"})
    await wait_for(lambda e: e["kind"] == "user" and "ชู 2 นิ้ว" in e["text"])
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
    cam = [c for c in info if c["source"] == "camera"]
    assert cam and "แดง" in cam[0]["snippet"], cam
    jobs = [e for _, e in events if e["kind"] == "team"]
    runs = {e["id"]: e["worker"] for e in jobs if e["state"] == "run"}
    assert "docs" in runs.values() and all(any(e["id"] == i and e["state"] != "run" for e in jobs) for i in runs), jobs
    assert "boss" in team and {"web", "docs", "camera"} <= set(team["workers"]), team
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
    assert len(users) == 6 and users[0].startswith("(ระบบ:"), f"expected greeting, docs, camera, barge-in, gesture, chat; got {users}"
    print("OK turns:", users)


asyncio.run(main())
