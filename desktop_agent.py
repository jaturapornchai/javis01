"""Computer-control worker (ลูกทีมคุมคอมพิวเตอร์): a short LLM loop over the agent-desktop CLI (Windows UI Automation).

Scope set by ลุงจืด (2026-11-04): look at the screen, open programs that are needed, press buttons that are needed,
ask first when unsure. So there is no typing tool at all, and a risky button or an unlisted program is held until
ลุงจืด says yes. That yes is checked in code (voice_agent -> confirm()), never taken from the LLM's own word.
Screen contents are never logged: snapshots may hold private text."""
import json
import logging
import os
import re
import subprocess
import time

import config as C

log = logging.getLogger("desktop_agent")
AD = os.path.join(C.BASE_DIR, "node_modules", "agent-desktop", "bin", "agent-desktop-win32-x64.exe")
# ponytail: a word list, not understanding. A matching (or nameless) button waits for ลุงจืด; upgrade: ask on every click.
RISKY = re.compile(r"ส่ง|ลบ|ยืนยัน|ตกลง|ใช่|ชำระ|จ่าย|โอน|ซื้อ|สั่ง|บันทึก|ติดตั้ง|ถอน|ออกจากระบบ|ปิดเครื่อง|รีสตาร์ท|"
                   r"send|delete|remove|submit|confirm|pay|buy|purchase|order|save|publish|post|transfer|install|"
                   r"format|sign ?out|log ?out|shut ?down|restart|accept|allow|agree|^ok$|^yes$", re.I)
CONFIRM_TTL = 120  # seconds a held action waits for ลุงจืด's yes
state = {"refs": {}, "pending": None}  # refs: "e12" -> (full ref, role, name) from the latest look()


def ad(*args):
    """One agent-desktop command -> its JSON envelope (never raises; no shell, args go straight to the exe)."""
    try:
        out = subprocess.run([AD, *args], capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
        return json.loads(out)
    except Exception as e:  # missing exe, timeout, bad JSON
        return {"ok": False, "error": {"code": type(e).__name__, "message": str(e)}}


def err(r):
    e = r.get("error") or {}
    return f"ไม่สำเร็จ ({e.get('code')}: {str(e.get('message', ''))[:150]})"


def flatten(node, out, depth=0):
    """Tree -> one line per named/addressable element; refs shortened to e12 so the LLM cannot garble them."""
    name, ref = (node.get("name") or "").strip(), node.get("ref_id")
    if ref or name or node.get("value"):
        short = ref.rsplit(":", 1)[1] if ref else ""
        if ref:
            state["refs"][short] = (ref, node.get("role", ""), name)
        value = f" = {str(node['value'])[:80]}" if node.get("value") else ""
        out.append(f"{'  ' * min(depth, 6)}{short + ' ' if short else ''}{node.get('role', '')} \"{name[:80]}\"{value}")
    for child in node.get("children") or []:
        flatten(child, out, depth + 1)


def list_apps():
    r = ad("list-apps")
    return "โปรแกรมที่เปิดอยู่: " + ", ".join(sorted({a["name"] for a in r["data"]["apps"]})) if r.get("ok") else err(r)


def look(app):
    r = ad("snapshot", "--app", str(app), "--compact")
    if not r.get("ok"):
        return err(r)
    state["refs"].clear()
    lines = []
    flatten(r["data"].get("tree") or {}, lines)
    return "\n".join(lines)[: C.TOOL_RESULT_CHARS] or "หน้าต่างนี้ไม่มีข้อความให้อ่าน"


def hold(kind, target, label):
    state["pending"] = {"kind": kind, "target": target, "label": label, "t": time.time()}
    return f"ยังไม่ได้ทำ: ต้องให้{C.USER_NAME}ยืนยันก่อนจะ{label} หยุดแล้วรายงานว่ารอยืนยัน"


def do(kind, target):
    r = ad(kind, target)
    log.info("desktop %s ok=%s", kind, r.get("ok"))
    return "ทำแล้ว" if r.get("ok") else err(r)


def launch(app):
    app = str(app).strip()
    if os.path.splitext(os.path.basename(app.lower()))[0] not in C.DESKTOP_SAFE_APPS:
        return hold("launch", app, f"เปิดโปรแกรม {app}")
    return do("launch", app)


def click(ref):
    hit = state["refs"].get(str(ref).strip().rsplit(":", 1)[-1].lstrip("@"))
    if not hit:
        return "ไม่รู้จัก ref นี้ ให้ look ก่อนแล้วเลือก ref จากผลล่าสุด"
    full, role, name = hit
    if not name or RISKY.search(name):
        return hold("click", full, f"กดปุ่ม \"{name or role}\"")
    return do("click", full)


def confirm():
    """ลุงจืด said yes (checked by the caller): run the held action. None if nothing is waiting or it expired."""
    p, state["pending"] = state["pending"], None
    if not p or time.time() - p["t"] > CONFIRM_TTL:
        return None
    return f"{p['label']}: {do(p['kind'], p['target'])}"


TOOLS = {"list_apps": list_apps, "look": look, "launch": launch, "click": click}


def _spec(name, desc, arg=None, arg_desc=""):
    props = {arg: {"type": "string", "description": arg_desc}} if arg else {}
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": [arg] if arg else []}}}


SPECS = [
    _spec("list_apps", "ดูรายชื่อโปรแกรมที่เปิดอยู่"),
    _spec("look", "อ่านข้อความและปุ่มในหน้าต่างของโปรแกรม ได้ ref ของปุ่มไว้กด", "app", "ชื่อโปรแกรมจาก list_apps เช่น chrome.exe"),
    _spec("launch", "เปิดโปรแกรม", "app", "ชื่อโปรแกรม เช่น notepad calc excel"),
    _spec("click", "กดปุ่มจาก ref ที่ได้จาก look ล่าสุด", "ref", "เช่น e12"),
]
PROMPT = (
    f"คุณคือลูกทีมผู้ดูและควบคุมคอมพิวเตอร์วินโดวส์ของ{C.USER_NAME} ทำงานที่หัวหน้าส่งมาด้วยเครื่องมือ list_apps look launch click "
    "เริ่มจาก list_apps หรือ look เสมอ ห้ามเดาว่าบนจอมีอะไร รายงานเฉพาะสิ่งที่อ่านได้จากผล look "
    "ไม่มีเครื่องมือพิมพ์ข้อความ ถ้างานต้องพิมพ์ให้รายงานว่าทำเองไม่ได้ "
    "ถ้าเครื่องมือตอบว่าต้องยืนยันก่อน ให้หยุดทันทีแล้วรายงานว่ารอยืนยันอะไร "
    "จบด้วยสรุปสั้นๆ ภาษาไทยว่าเห็นอะไร ทำอะไรไปแล้ว และอะไรยังไม่ได้ทำ"
)


def desktop_task(task, chat, steps=6):
    """Worker loop: `chat(messages, tools)` returns (text, calls as {"id", "name", "args"}), OpenAI-style messages.
    Returns (cards, text) like every tool."""
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": str(task)[:500]}]
    summary = ""
    for i in range(steps + 1):
        text, calls = chat(messages, SPECS if i < steps and not state["pending"] else None)
        if not calls:
            summary = text.strip()
            break
        messages.append({"role": "assistant", "content": text, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
            for c in calls]})
        for c in calls:
            fn = TOOLS.get(c["name"])
            try:
                out = fn(**c["args"]) if fn else "ไม่มีเครื่องมือนี้"
            except TypeError as e:  # wrong argument names from the LLM
                out = f"อาร์กิวเมนต์ไม่ถูกต้อง ({e})"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    if p := state["pending"]:
        summary += f"\n(รอ{C.USER_NAME}ยืนยันก่อน{p['label']})"
    summary = summary.strip() or "ลูกทีมคุมคอมพิวเตอร์ไม่ได้รายงานผล"
    return [{"source": "desktop", "title": f"คอมพิวเตอร์: {str(task)[:60]}", "snippet": summary[:300], "url": None}], summary


if __name__ == "__main__":  # self-check with a fake exe and a scripted LLM: no real clicks
    sent = []
    tree = {"children": [{"name": "Untitled - Notepad", "children": [
        {"ref_id": "@sX:e1", "role": "button", "name": "Help"}, {"ref_id": "@sX:e2", "role": "button", "name": "Save"},
        {"role": "text", "name": "สวัสดี"}]}]}

    def fake_ad(*args):
        sent.append(args)
        if args[0] == "snapshot":
            return {"ok": True, "data": {"tree": tree}}
        return {"ok": True, "data": {"apps": [{"name": "notepad.exe"}]}} if args[0] == "list-apps" else {"ok": True}
    ad = fake_ad  # noqa: F811  (module-level rebinding, so every tool uses the fake)

    def call(name, **args):
        return {"id": f"call_{name}", "name": name, "args": args}
    script = iter([[call("look", app="notepad.exe")], [call("click", ref="e1"), call("click", ref="@sX:e2")], []])

    def fake_chat(messages, tools):
        calls = next(script)
        return ("" if calls else "เห็น Notepad กด Help แล้ว รอยืนยันกด Save"), calls
    cards, text = desktop_task("เปิดดู notepad แล้วกดบันทึก", fake_chat)
    assert "e2 button \"Save\"" in look("notepad.exe") and "สวัสดี" in look("notepad.exe")
    assert ("click", "@sX:e1") in sent and ("click", "@sX:e2") not in sent, sent  # Help pressed, Save held
    assert state["pending"]["target"] == "@sX:e2" and "รอลุงจืดยืนยัน" in text.replace(C.USER_NAME, "ลุงจืด"), text
    assert confirm() == 'กดปุ่ม "Save": ทำแล้ว' and ("click", "@sX:e2") in sent and confirm() is None
    assert launch("calc") == "ทำแล้ว" and "ยืนยัน" in launch("C:/tmp/evil.exe") and state["pending"]["kind"] == "launch"
    state["pending"]["t"] -= CONFIRM_TTL + 1
    assert confirm() is None, "an old held action must expire"
    assert click("e99").startswith("ไม่รู้จัก") and cards[0]["source"] == "desktop"
    print("desktop_agent self-test OK")
