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
}
KB_CALLS = {"ถามระบบ"}
NOT_FOUND = re.compile(r"ไม่พบ|ไม่เจอ|ยังไม่มีข้อมูล")
CODE_TALK = re.compile(r"ฟังก์ชัน|ฟิลด์|\b[a-z]+[A-Z]\w*|\w+_\w+")  # 2026-11-08 04:09 a caller heard "ฟังก์ชัน FIFO"
NAME_ASK = re.compile(r"ขอทราบชื่อ|ขอชื่อ|ถามชื่อ|ชื่ออะไร|(เรียนสาย|พูด)กับใคร")  # ponytail: wordings seen in eval runs

cuts = []


class Grab(logging.Handler):  # the server's cut lines, shown under the turn they belong to
    def emit(self, record):
        if (msg := record.getMessage()).startswith("cut") or record.levelno >= logging.WARNING:  # a silent turn says why
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
    extra = va.PUBLIC["prompt"].removeprefix(va.BASE_PROMPT) + "\n" + va.OWN_WORDS
    leaked = [line for caller, line in MUST_CUT if va.judge(f"{caller}\n{extra}", line, "")]
    missed = [line for caller, line in SHOULD_PASS if not va.judge(f"{caller}\n{extra}", line, "")]
    print(f"ตัวตรวจ: ต้องตัด {len(MUST_CUT) - len(leaked)}/{len(MUST_CUT)}, คุยเล่นผ่าน {len(SHOULD_PASS) - len(missed)}/{len(SHOULD_PASS)}")
    return [f"ตัวตรวจปล่อยประโยคแต่ง: {x}" for x in leaked] + ([f"ตัวตรวจตัดคุยเล่น: {x}" for x in missed] if len(missed) > 2 else [])


def run_call(name, lines):
    """One scripted call, turn by turn on one history; returns what went wrong."""
    va.SESSIONS[""], bad = [], []
    for _ in va.greet_public():  # every real call opens with the secretary's hello
        pass
    print(f"\n== {name}\n  เลขาทัก: {va.SESSIONS[''][-1]['content'] if va.SESSIONS[''] else '(ไม่ได้ทัก)'}")
    for line in lines:
        cuts.clear()
        va.early_words = lambda data, line=line: line
        try:
            for _ in va.reply_public((48000, np.zeros(48000, np.int16))):
                pass
        except Exception as e:  # any crash fails the gate
            bad.append(f"{name}: {line}: {type(e).__name__} {e}")
            continue
        said, _, panel = va.SESSIONS[""][-1]["content"].partition(C.SCREEN_MARK)
        said, panel = said.strip(), panel.strip()
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
                    ("คำถามระบบที่มีในคลังแต่ตอบว่าไม่พบ", name in KB_CALLS and bool(NOT_FOUND.search(said)))]
        bad += [f"{name}: {line}: {p}" for p, hit in problems if hit]
    asks = sum(bool(NAME_ASK.search(m["content"])) for m in va.SESSIONS[""] if m["role"] == "assistant")
    return bad + ([f"{name}: ถามชื่อ {asks} ครั้งในสายเดียว"] if asks > 2 else [])


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
