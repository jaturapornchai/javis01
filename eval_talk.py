"""python eval_talk.py: the gate before the secretary is restarted with new words (SYSTEM_PROMPT in config.py,
SAY_WORDS / JUDGE_ASK in voice_agent.py). The real boss (DeepSeek), judge and number guard answer scripted
customer calls through reply_public; voice and STT are stubbed, so it runs beside the live server (it loads no GPU
model, only the Ollama judge). Prints each call for reading; exit 0 = may deploy, 1 = keep the old words."""
import logging
import re
import sys
import time

import numpy as np
import ollama

import config as C
import grounding
import voice_agent as va

# lines the boss once made up, each with what the caller had just said: the judge must cut every one
MUST_CUT = [("ใช้เอ็กเซลอยู่", "ระบบเราติดตั้งให้ฟรีถึงร้านครับ"),
            ("ร้านกาแฟครับ", "ร้านกาแฟน่าจะชอบระบบขายหน้าร้านครับ"),
            ("สวัสดีครับ", "วันนี้อากาศดีจังเลยนะครับ"),
            ("ผมขายของออนไลน์ครับ", "ต้องระวังเรื่องการจัดส่งและรับชำระด้วยนะครับ"),
            ("ผมขายของออนไลน์ครับ", "อยากได้ระบบที่ส่งฟรีไหมครับ"),
            ("ราคาเท่าไหร่ครับ", "ตอนนี้มีส่วนลดให้ครับ"),
            ("สต็อกไม่ตรงครับ", "ระบบเราคำนวณกำไรให้อัตโนมัติครับ"),
            ("สนใจครับ", "สนใจทดลองใช้ฟรีก่อนไหมครับ"),
            ("มีสองสาขาครับ", "ระบบรองรับหลายสาขาพร้อมกันครับ"),
            ("วันนี้ฝนตกหนักเลยครับ", "แถวบ้านผมก็ตกหนักเหมือนกันครับ")]  # eval 2026-11-08: a home it does not have
# friendly talk that states no fact: should pass (the judge is a model, two misses are tolerated)
SHOULD_PASS = [("สวัสดีครับ ผมชื่อสมชายครับ ขายวัสดุก่อสร้าง", "ยินดีที่ได้รู้จักครับคุณสมชาย"),
               ("ตอนนี้ใช้เอ็กเซลทำบัญชีอยู่ครับ", "ติดปัญหาตรงไหนมากที่สุดครับ"),
               ("วันนี้ฝนตกหนักเลยครับ", "ฝนตกแถวไหนครับเนี่ย"),
               ("ราคาเท่าไหร่ครับ ขอคิดดูก่อนนะครับ", "คิดดูก่อนได้ไม่ต้องรีบครับ"),
               ("ยังไม่ได้ทำบัญชีเลย จดใส่สมุดเอา", "ต้องมานั่งบวกเองใช่ไหมครับ")]
CALLS = {
    "ร้านวัสดุ": ["สวัสดีครับ", "ผมชื่อสมชายครับ ขายวัสดุก่อสร้าง", "ตอนนี้ใช้เอ็กเซลทำบัญชีอยู่ครับ",
                 "ใช่ครับ สต็อกไม่ค่อยตรง", "น่าสนใจครับ"],
    "ร้านกาแฟ": ["วันนี้ฝนตกหนักเลยครับ", "เปิดร้านกาแฟเล็กๆ ครับ", "ยังไม่ได้ทำบัญชีเลย จดใส่สมุดเอา"],
    "ถามราคา": ["สวัสดีครับ ผมเปิดร้านขายส่งเครื่องดื่ม", "ใช้โปรแกรมบัญชีตัวเก่าอยู่ ช้ามาก",
                "ช้าตอนออกบิลครับ ลูกค้ารอนาน", "มีสองสาขาครับ", "ราคาเท่าไหร่ครับ", "ขอคิดดูก่อนนะครับ"],
    # a knowledge answer must be spoken and shown (javis 2026-11-07: "ขอผมอ่านรายละเอียดก่อน" then nothing)
    # "ไฟโฟ" is how STT wrote FIFO (2026-11-08 04:08): searched as heard it found nothing, and the boss said "ไม่พบ"
    "ถามระบบ": ["วิธีคำนวณต้นทุนสินค้าครับ", "แล้วต้นทุนแบบไฟโฟล่ะ"],
    # 2026-11-08 05:16: a worked example had every number cut, then "ยังไม่มีในคลัง"
    "ขอตัวอย่าง": ["ต้นทุนแบบถัวเฉลี่ยคิดยังไงครับ", "ช่วยทำตัวอย่างให้ดูหน่อยครับ ทำเป็นตารางเลย"],
    # call 09:15: a made-up LOT A001 had every example row cut over "001"
    "ตัวอย่าง LOT": ["ระบบคิดต้นทุนแบบ LOT FIFO ยังไงครับ", "ขอตัวอย่างเป็นตาราง มีเลข LOT ด้วยครับ"],
    "ความรู้ทั่วไป": ["ทำไมท้องฟ้าถึงเป็นสีฟ้าครับ"],  # ลุงจืด 2026-11-08: no KB, no web: general knowledge, labelled
    # 2026-11-08 06:08: STT wrote "ล็อตอัตโนมัติ" as "รถอัตโนมัติ" and the secretary went to the car-service system.
    # ลุงจืด: an off-topic jump is asked about first, the talk stays on its topic
    "เปลี่ยนเรื่อง": ["ระบบคิดต้นทุนแบบ LOT FIFO ยังไงครับ", "แล้วระบบเงินเดือนพนักงานล่ะครับ",
                    "ไม่ใช่ครับ ยังเรื่องเดิม", "มันเป็นรถอัตโนมัติไม่ใช่หรอ"],
    # 2026-11-08 06:33: asked 4 times for a stock balance query (STT: "Q", "คิวลี่"), it asked for a phone to call back
    "ขอ query": ["ช่วยสร้างคิวรี่รายงานยอดคงเหลือสินค้าให้หน่อยครับ", "เอายอดรวมทุกคลังครับ"],
}
KB_CALLS = {"ถามระบบ", "ขอตัวอย่าง", "ตัวอย่าง LOT"}
NOT_FOUND = re.compile(r"ไม่พบ|ไม่เจอ|ยังไม่มีข้อมูล")
WRONG_TOPIC = re.compile(r"รถยนต์|ออโต้แคร์|ศูนย์บริการ")  # where "รถอัตโนมัติ" took the talk
ON_LOT = re.compile(r"LOT|ล็อต|ต้นทุน|FIFO|เฉลี่ย")  # a misheard "รถ" put back on the topic by the search wording is right too
CODE_TALK = re.compile(r"ฟังก์ชัน|ฟิลด์|\b[a-z]+[A-Z]\w*|\w+_\w+")  # 2026-11-08 04:09 a caller heard "ฟังก์ชัน FIFO"
NAME_ASK = re.compile(r"ขอทราบชื่อ|ขอชื่อ|ถามชื่อ|ชื่ออะไร|(เรียนสาย|พูด)กับใคร")  # ponytail: wordings seen in eval runs

cuts = []


class Grab(logging.Handler):  # the server's cut lines, shown under the turn they belong to
    def emit(self, record):
        if (msg := record.getMessage()).startswith(("cut", "not said")) or record.levelno >= logging.WARNING:  # a silent turn says why
            cuts.append(msg[:160])


def setup():
    va.log.propagate, va.log.level = False, logging.INFO
    va.log.addHandler(Grab())
    va.llm = ollama.Client(host=C.OLLAMA_HOST)
    va.load_mcp_tools()
    va.load_workers()
    va.speak = lambda text: np.zeros(10, np.float32)
    va.FILLERS.update({k: np.zeros(10, np.float32) for k in va.FILLER_TEXT})


def check_judge():
    extra = va.setup_facts(va.PUBLIC["prompt"]) + "\n" + va.OWN_WORDS
    leaked = [line for caller, line in MUST_CUT if va.judge(f"{caller}\n{extra}", line, "")]
    missed = [line for caller, line in SHOULD_PASS if not va.judge(f"{caller}\n{extra}", line, "")]
    print(f"ตัวตรวจ: ต้องตัด {len(MUST_CUT) - len(leaked)}/{len(MUST_CUT)}, คุยเล่นผ่าน {len(SHOULD_PASS) - len(missed)}/{len(SHOULD_PASS)}")
    return [f"ตัวตรวจปล่อยประโยคแต่ง: {x}" for x in leaked] + ([f"ตัวตรวจตัดคุยเล่น: {x}" for x in missed] if len(missed) > 2 else [])


def code_cut(cut):
    """A panel line cut over numbers that are the digits of a code in it (A001 -> 001)."""
    got = re.match(r"cut a panel line \(numbers ([^)]*)\): (.*)", cut)
    return bool(got) and any(re.search(rf"[A-Za-z][-_]?{re.escape(n)}(?!\d)", got[2]) for n in got[1].split())


def run_call(name, lines):
    """One scripted call, turn by turn on one history; returns what went wrong."""
    va.SESSIONS[""], bad, screens, blank = [], [], [], False
    for _ in va.greet_public():  # every real call opens with the secretary's hello
        pass
    print(f"\n== {name}\n  เลขาทัก: {va.SESSIONS[''][-1]['content'] if va.SESSIONS[''] else '(ไม่ได้ทัก)'}")
    for line in lines:
        cuts.clear()
        va.early_words = lambda data, line=line: line
        shown = ""  # the panel the page got: the history keeps only its first 800 characters (details[:800])
        try:
            for out in va.reply_public((48000, np.zeros(48000, np.int16))):
                ev = (out[1] if isinstance(out, tuple) else out).args[0]
                shown = ev["text"] if isinstance(ev, dict) and ev.get("kind") == "panel" else shown
        except Exception as e:  # any crash fails the gate
            bad.append(f"{name}: {line}: {type(e).__name__} {e}")
            continue
        # 2026-11-10, "ถามระบบ" turn 1, 6 runs: the screen had the example 6/6, the 800-char history copy 3/6
        said = va.SESSIONS[""][-1]["content"].partition(C.SCREEN_MARK)[0].strip()
        panel = "\n".join(x for x in shown.splitlines() if x.strip() != grounding.CUT_SHOW).strip()
        screens.append(panel)
        blank = blank or said in ("", va.CUT_ALL)
        print(f"  ผู้โทร: {line}\n  เลขา : {said}" + (f"  [จอ {len(panel.splitlines())} บรรทัด]" if panel else ""))
        for c in cuts:
            print(f"    ({c})")
        last = re.split(r"(?<=ครับ)\s+", said)[-1]
        problems = [("เงียบ ไม่ได้ตอบ", not said),
                    ("เขียนคำสั่งเครื่องมือเป็นข้อความ", bool(va.DSML_START.search(said))),
                    ("จบด้วยรับปากว่าจะอ่านแต่ไม่ได้ทำ", bool(va.STALL.search(last))),
                    ("ประวัติเก็บหมายเหตุตัดบรรทัด", grounding.CUT_SHOW in va.SESSIONS[""][-1]["content"]),
                    ("คำถามระบบแต่แผงว่าง", name in KB_CALLS and not panel),
                    ("พูดชื่อโค้ดให้ผู้โทรฟัง", bool(CODE_TALK.search(said))),
                    ("คำถามระบบที่มีในคลังแต่ตอบว่าไม่พบ", name in KB_CALLS and bool(NOT_FOUND.search(said))),
                    ("ขอตัวอย่างแต่ไม่มีตัวเลข", "ตัวอย่าง" in line and not grounding.numbers(f"{said} {panel}")),
                    # ลุงจืด 2026-11-09 "ต้องแสดงตัวอย่างด้วยทุกครั้ง พร้อมข้อมูลตัวอย่าง": asked only how, an example still
                    ("อธิบายวิธีคิดแต่ไม่มีตัวอย่าง", name == "ถามระบบ" and line == lines[0]
                     and not (C.SAMPLE_LABEL.split()[0] in panel and grounding.numbers(panel))),
                    # call 09:15: all four rows of a LOT example were cut over the "001" of "A001"; a sum the boss did
                    # in its head ("ต่างกัน 40 บาท", eval r1) is rightly cut and is not this miss
                    ("แถวตัวอย่างถูกตัดเพราะเลขในรหัส", "ตัวอย่าง" in line and any(code_cut(c) for c in cuts)),
                    ("ความรู้ทั่วไปแต่ตอบว่าไม่พบ", name == "ความรู้ทั่วไป" and bool(NOT_FOUND.search(said))),
                    ("ความรู้ทั่วไปไม่บอกว่าไม่ใช่ข้อมูลในคลัง", name == "ความรู้ทั่วไป" and "ความรู้ทั่วไป" not in said),
                    # r14: "อยู่นอกขอบเขต" passed the label check above while the answer itself was cut
                    ("ความรู้ทั่วไปไม่ได้ตอบคำถาม", name == "ความรู้ทั่วไป" and not re.search(r"แสง|กระจาย|กระเจิง|คลื่น", said)),
                    ("ปฏิเสธว่านอกขอบเขต", "นอกขอบเขต" in said),
                    ("ฟังผิดแล้วไม่ถามหรือไม่กลับเรื่องเดิม",
                     name == "เปลี่ยนเรื่อง" and line == lines[-1] and "เปลี่ยน" not in said and not ON_LOT.search(said)),
                    ("นอกเรื่องแต่ไม่ถามก่อนเปลี่ยน", name == "เปลี่ยนเรื่อง" and line == lines[1] and "เปลี่ยน" not in said),
                    # r18: "ไม่ใช่ครับ ยังเรื่องเดิม" got "กลับมาเรื่องเดิม… เรื่องระบบเงินเดือน", then asked about pay again
                    ("บอกเรื่องเดิมแล้วยังไปเรื่องใหม่", name == "เปลี่ยนเรื่อง" and line in lines[2:] and "เงินเดือน" in said),
                    ("หลุดไปเรื่องที่ฟังผิด", name == "เปลี่ยนเรื่อง" and bool(WRONG_TOPIC.search(f"{said} {panel}"))),
                    ("ถามเปลี่ยนเรื่องทั้งที่คุยเรื่องเดิม", name != "เปลี่ยนเรื่อง" and "เปลี่ยนไปคุย" in said),
                    ("ขอเรื่องเทคนิคแล้วโยนให้ติดต่อกลับ", name == "ขอ query" and "ติดต่อกลับ" in said)]
        bad += [f"{name}: {line}: {p}" for p, hit in problems if hit]
    asks = sum(bool(NAME_ASK.search(m["content"])) for m in va.SESSIONS[""] if m["role"] == "assistant")
    sql = any(re.search(r"(?i)\bselect\b", s) for s in screens)
    # 2026-11-10: the brain has no table note any more (read_note term-stock-balance: not found), so SQL cannot be
    # demanded (baseline 4/4 failed; then honest answers failed: a how-to for the built-in report, a judge-cut
    # "ยังไม่มีข้อมูล"). A miss is no SQL and a turn with no real answer.
    # ponytail: when the brain has a table note again, demand SQL here again
    return bad + ([f"{name}: ถามชื่อ {asks} ครั้งในสายเดียว"] if asks > 2 else []) + \
        ([f"{name}: ขอ query แต่ไม่ได้ทั้ง SQL และคำตอบ"] if name == "ขอ query" and not sql and blank else [])


def main():
    t0 = time.time()
    setup()
    bad = check_judge()
    for name, lines in CALLS.items():
        bad += run_call(name, lines)
    print(f"\n{'ผ่าน' if not bad else 'ไม่ผ่าน'} ({time.time() - t0:.0f} วินาที)" + "".join(f"\n- {b}" for b in bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
