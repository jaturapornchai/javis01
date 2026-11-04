"""Read-only Google tools (Gmail / Calendar / Drive) for the voice agent LLM.
Each tool returns (cards, text). Never logs tokens, headers or email bodies."""
import base64
import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

import config as C

log = logging.getLogger("google_tools")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
CLIENT_SECRET_PATH = os.path.join(C.BASE_DIR, "client_secret.json")
TOKEN_PATH = os.path.join(C.BASE_DIR, ".secrets", "google_token.json")

http = httpx.Client(timeout=10)
TZ = ZoneInfo("Asia/Bangkok")
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
CAL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
DRIVE = "https://www.googleapis.com/drive/v3/files"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
TEXT_CAP, BODY_CAP, PDF_MAX = 6000, 4000, 20 * 1024 * 1024
DAYS_TH = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
MONTHS_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

_creds_cache = None


def _creds():
    global _creds_cache
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    if _creds_cache is None:
        _creds_cache = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not _creds_cache.valid and _creds_cache.refresh_token:
        _creds_cache.refresh(Request())
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(_creds_cache.to_json())
    return _creds_cache


def connected() -> bool:
    try:
        return bool(_creds())
    except Exception:
        return False


def _token() -> str:
    return _creds().token


def _get(url, **params):
    r = http.get(url, params=params, headers={"Authorization": f"Bearer {_token()}"})
    r.raise_for_status()
    return r


def _clean(s):
    if not isinstance(s, str):
        raise ValueError("query")
    return s.strip()[:200]


def _id(s):
    if not isinstance(s, str) or not ID_RE.match(s.strip()):
        raise ValueError("id")
    return s.strip()


def _safe(name):
    """Wrap a tool: log only name + error type, return Thai message on failure."""
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except ValueError:
                return [], "ข้อมูลที่ส่งมาไม่ถูกต้อง"
            except FileNotFoundError:
                return [], "ยังไม่ได้เชื่อม Google (ให้รัน google_login.py ก่อน)"
            except Exception as e:
                log.warning("%s failed: %s", name, type(e).__name__)
                return [], "เรียกใช้ Google ไม่สำเร็จ ลองใหม่อีกครั้ง"
        run.__name__ = fn.__name__
        return run
    return deco


def _out(cards, items, empty):
    if not items:
        return [], empty
    return cards, json.dumps(items, ensure_ascii=False, separators=(",", ":"))[:TEXT_CAP]


def _card(source, title, snippet, url):
    return {"source": source, "title": title or "(ไม่มีชื่อ)", "snippet": (snippet or "")[:300], "url": url}


# ---------------- Gmail ----------------
@_safe("gmail_search")
def gmail_search(query):
    q = _clean(query)
    ids = _get(f"{GMAIL}/messages", q=q, maxResults=5).json().get("messages", [])

    def meta(m):
        return _get(f"{GMAIL}/messages/{m['id']}", format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]).json()
    # ponytail: Gmail's multipart batch endpoint is the upgrade (1 request instead of 5)
    with ThreadPoolExecutor(max_workers=5) as ex:
        msgs = list(ex.map(meta, ids))
    items, cards = [], []
    for m in msgs:
        h = {x["name"].lower(): x["value"] for x in m.get("payload", {}).get("headers", [])}
        items.append({"id": m["id"], "from": h.get("from", ""), "subject": h.get("subject", ""),
                      "date": h.get("date", ""), "snippet": m.get("snippet", "")})
        cards.append(_card("gmail", h.get("subject"), f'{h.get("from", "")} — {m.get("snippet", "")}',
                           f"https://mail.google.com/mail/u/0/#all/{m.get('threadId', m['id'])}"))
    return _out(cards, items, "ไม่พบอีเมลที่ตรงกับคำค้น")


def _b64(data):
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")


def _parts(p, mime):
    if p.get("mimeType") == mime and p.get("body", {}).get("data"):
        yield _b64(p["body"]["data"])
    for c in p.get("parts", []) or []:
        yield from _parts(c, mime)


@_safe("gmail_read")
def gmail_read(message_id):
    mid = _id(message_id)
    m = _get(f"{GMAIL}/messages/{mid}", format="full").json()
    p = m.get("payload", {})
    h = {x["name"].lower(): x["value"] for x in p.get("headers", [])}
    body = "\n".join(_parts(p, "text/plain"))
    if not body.strip():
        html = "\n".join(_parts(p, "text/html"))
        body = re.sub(r"\s+", " ", re.sub(r"(?is)<(script|style).*?</\1>|<[^>]+>", " ", html))
    body = body.strip()[:BODY_CAP]
    item = {"id": mid, "subject": h.get("subject", ""), "from": h.get("from", ""),
            "date": h.get("date", ""), "body": body}
    if not body and not item["subject"]:
        return [], "ไม่พบเนื้อหาอีเมลนี้"
    card = _card("gmail", h.get("subject"), body, f"https://mail.google.com/mail/u/0/#all/{m.get('threadId', mid)}")
    return [card], json.dumps(item, ensure_ascii=False, separators=(",", ":"))


# ---------------- Calendar ----------------
def _events(tmin, tmax, q="", n=10):
    p = dict(timeMin=tmin.isoformat(), timeMax=tmax.isoformat(), singleEvents="true",
             orderBy="startTime", maxResults=n)
    if q:
        p["q"] = q
    return _get(CAL, **p).json().get("items", [])


def _when(ev):
    s, e = ev.get("start", {}), ev.get("end", {})
    if "date" in s:
        d = datetime.fromisoformat(s["date"])
        return f"{DAYS_TH[d.weekday()]} {d.day} {MONTHS_TH[d.month - 1]} ทั้งวัน"
    a = datetime.fromisoformat(s["dateTime"]).astimezone(TZ)
    b = datetime.fromisoformat(e["dateTime"]).astimezone(TZ) if "dateTime" in e else a
    return f"{DAYS_TH[a.weekday()]} {a.day} {MONTHS_TH[a.month - 1]} {a:%H:%M}-{b:%H:%M}"


@_safe("calendar_events")
def calendar_events(days=1, query=""):
    days = int(days)
    if not 1 <= days <= 31:
        raise ValueError("days")
    q = _clean(query) if query else ""
    now = datetime.now(timezone.utc)
    items, cards = [], []
    for ev in _events(now, now + timedelta(days=days), q):
        it = {"id": ev.get("id"), "title": ev.get("summary", "(ไม่มีชื่อ)"), "when": _when(ev)}
        if ev.get("location"):
            it["location"] = ev["location"]
        if ev.get("attendees"):
            it["attendees"] = len(ev["attendees"])
        items.append(it)
        cards.append(_card("calendar", it["title"],
                           " · ".join([it["when"]] + ([it["location"]] if "location" in it else [])),
                           ev.get("htmlLink")))
    return _out(cards, items, "ไม่พบนัดหมายในช่วงเวลานี้")


def upcoming_events(minutes):
    """Events starting within the next N minutes (for spoken reminders). Not an LLM tool."""
    try:
        now = datetime.now(timezone.utc)
        out = []
        for ev in _events(now, now + timedelta(minutes=int(minutes)), n=20):
            start = ev.get("start", {}).get("dateTime")
            if start and datetime.fromisoformat(start) > now:  # timeMin matches on end time: skip events already running
                out.append({"id": ev["id"], "summary": ev.get("summary", ""), "start": datetime.fromisoformat(start)})
        return out
    except Exception as e:
        log.warning("upcoming_events failed: %s", type(e).__name__)
        return []


# ---------------- Drive ----------------
def _drive_q(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


@_safe("drive_search")
def drive_search(query):
    q = _clean(query)
    # no orderBy: Drive rejects it together with fullText (results come in relevance order)
    files = _get(DRIVE, q=f"fullText contains '{_drive_q(q)}' and trashed=false", pageSize=5,
                 fields="files(id,name,mimeType,modifiedTime,webViewLink)").json().get("files", [])
    items = [{"id": f["id"], "name": f["name"], "type": f.get("mimeType", ""),
              "modified": f.get("modifiedTime", "")} for f in files]
    cards = [_card("drive", f["name"], f"แก้ไขล่าสุด {f.get('modifiedTime', '')[:10]}", f.get("webViewLink"))
             for f in files]
    return _out(cards, items, "ไม่พบไฟล์ใน Drive ที่ตรงกับคำค้น")


EXPORT = {"application/vnd.google-apps.document": "text/plain",
          "application/vnd.google-apps.spreadsheet": "text/csv",
          "application/vnd.google-apps.presentation": "text/plain"}


@_safe("drive_read")
def drive_read(file_id):
    fid = _id(file_id)
    meta = _get(f"{DRIVE}/{fid}", fields="id,name,mimeType,size,webViewLink").json()
    mt = meta.get("mimeType", "")
    if mt not in EXPORT and int(meta.get("size", 0)) > PDF_MAX:  # any downloaded file, not only PDF
        return [], "ไฟล์ใหญ่เกิน 20 MB อ่านไม่ได้"
    if mt in EXPORT:
        text = _get(f"{DRIVE}/{fid}/export", mimeType=EXPORT[mt]).text
    elif mt == "application/pdf":
        from pypdf import PdfReader
        raw = _get(f"{DRIVE}/{fid}", alt="media").content
        if len(raw) > PDF_MAX:
            return [], "ไฟล์ PDF ใหญ่เกิน 20 MB อ่านไม่ได้"
        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(raw)).pages)
    elif mt.startswith("text/") or mt in ("application/json", "application/xml"):
        text = _get(f"{DRIVE}/{fid}", alt="media").text
    else:
        return [], "ไฟล์ชนิดนี้อ่านเป็นข้อความไม่ได้"
    text = text.strip()[:TEXT_CAP]
    if not text:
        return [], "ไม่พบเนื้อหาข้อความในไฟล์นี้"
    return [_card("drive", meta.get("name"), text, meta.get("webViewLink"))], \
        json.dumps({"name": meta.get("name"), "content": text}, ensure_ascii=False)


# ---------------- registry ----------------
TOOLS = {"gmail_search": gmail_search, "gmail_read": gmail_read, "calendar_events": calendar_events,
         "drive_search": drive_search, "drive_read": drive_read}
STATUS = {"gmail_search": "กำลังค้นอีเมล", "gmail_read": "กำลังอ่านอีเมล", "calendar_events": "กำลังดูปฏิทิน",
          "drive_search": "กำลังค้นไฟล์ใน Drive", "drive_read": "กำลังอ่านไฟล์ใน Drive"}


def _spec(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


def _S(d):
    return {"type": "string", "description": d}


TOOL_SPECS = [
    _spec("gmail_search", "ค้นอีเมลใน Gmail ใช้เมื่อผู้ใช้ถามถึงอีเมล/เมลใหม่/เมลจากใคร ใช้ไวยากรณ์ Gmail เช่น "
          "'is:unread newer_than:1d', 'from:สมชาย subject:ใบเสนอราคา' ได้ผลลัพธ์ 5 ฉบับพร้อม id",
          {"query": _S("คำค้นแบบ Gmail")}, ["query"]),
    _spec("gmail_read", "อ่านเนื้อหาอีเมลฉบับเต็มจาก id ที่ได้จาก gmail_search ใช้เมื่อผู้ใช้อยากรู้รายละเอียดในเมล",
          {"message_id": _S("id ของอีเมลจาก gmail_search")}, ["message_id"]),
    _spec("calendar_events", "ดูนัดหมายในปฏิทิน Google ตั้งแต่ตอนนี้ไปอีก N วัน ใช้เมื่อถามว่าวันนี้/พรุ่งนี้/สัปดาห์นี้มีนัดอะไร",
          {"days": {"type": "integer", "description": "จำนวนวันข้างหน้า 1-31 (ค่าเริ่มต้น 1)"},
           "query": _S("คำค้นในชื่อนัด (ไม่บังคับ)")}, []),
    _spec("drive_search", "ค้นไฟล์ใน Google Drive จากเนื้อหาหรือชื่อ ใช้เมื่อผู้ใช้ถามหาเอกสาร/ชีต/สไลด์/ไฟล์",
          {"query": _S("คำค้น")}, ["query"]),
    _spec("drive_read", "อ่านเนื้อหาไฟล์ใน Drive (Docs, Sheets, Slides, PDF, ข้อความ) จาก id ที่ได้จาก drive_search",
          {"file_id": _S("id ของไฟล์จาก drive_search")}, ["file_id"]),
]


if __name__ == "__main__":
    def b64(s):
        return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")

    seen = {}

    def handler(req):
        u, p = req.url.path, dict(req.url.params)
        assert req.headers["authorization"] == "Bearer T"
        if u.endswith("/messages"):
            return httpx.Response(200, json={"messages": [] if p["q"] == "none" else [{"id": "m1"}, {"id": "m2"}]})
        if u.endswith("/messages/m1") and p["format"] == "metadata":
            return httpx.Response(200, json={"id": "m1", "threadId": "t1", "snippet": "สวัสดี", "payload": {"headers": [
                {"name": "From", "value": "a@x.com"}, {"name": "Subject", "value": "ประชุม"}, {"name": "Date", "value": "d"}]}})  # mock Gmail header, not a real address  pragma: allowlist secret
        if u.endswith("/messages/m2") and p["format"] == "metadata":
            return httpx.Response(200, json={"id": "m2", "threadId": "t2", "snippet": "s", "payload": {"headers": []}})
        if u.endswith("/messages/m1"):
            return httpx.Response(200, json={"id": "m1", "threadId": "t1", "payload": {
                "headers": [{"name": "Subject", "value": "ประชุม"}], "mimeType": "multipart/alternative", "parts": [
                    {"mimeType": "text/html", "body": {"data": b64("<p>html</p>")}},
                    {"mimeType": "text/plain", "body": {"data": b64("เนื้อหา plain")}}]}})
        if u.endswith("/events"):
            if p.get("q") == "none":
                return httpx.Response(200, json={"items": []})
            return httpx.Response(200, json={"items": [
                {"id": "e1", "summary": "ประชุมทีม", "htmlLink": "https://cal/e1", "location": "ห้อง A",
                 "attendees": [{}, {}], "start": {"dateTime": "2026-11-04T14:00:00+07:00"},
                 "end": {"dateTime": "2026-11-04T15:00:00+07:00"}},
                {"id": "e2", "summary": "วันหยุด", "start": {"date": "2026-11-05"}, "end": {"date": "2026-11-06"}}]})
        if u.endswith("/files"):
            seen["q"] = p["q"]
            return httpx.Response(200, json={"files": [] if "none" in p["q"] else [
                {"id": "f1", "name": "แผน", "mimeType": "x", "modifiedTime": "2026-11-01T00:00:00Z", "webViewLink": "https://d/f1"}]})
        if u.endswith("/files/doc1"):
            return httpx.Response(200, json={"id": "doc1", "name": "Doc", "mimeType": "application/vnd.google-apps.document"})
        if u.endswith("/files/doc1/export"):
            assert p["mimeType"] == "text/plain"
            return httpx.Response(200, text="เนื้อเอกสาร")
        return httpx.Response(404)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    _token = lambda: "T"

    cards, text = gmail_search("is:unread")
    assert len(cards) == 2 and cards[0]["source"] == "gmail" and cards[0]["url"].endswith("#all/t1"), cards
    assert '"id":"m1"' in text and "ประชุม" in text
    assert gmail_search("none") == ([], "ไม่พบอีเมลที่ตรงกับคำค้น")
    cards, text = gmail_read("m1")
    assert "เนื้อหา plain" in text and "html" not in text and cards[0]["title"] == "ประชุม"
    assert gmail_read("../x") == ([], "ข้อมูลที่ส่งมาไม่ถูกต้อง")
    assert drive_read("../x")[0] == [] and gmail_search(5)[0] == []

    cards, text = calendar_events(2)
    assert len(cards) == 2 and "พุธ 4 พ.ย. 14:00-15:00" in text and "ทั้งวัน" in text and '"attendees":2' in text, text
    assert cards[0]["url"] == "https://cal/e1"
    assert calendar_events(0)[0] == [] and calendar_events(1, "none") == ([], "ไม่พบนัดหมายในช่วงเวลานี้")
    ue = upcoming_events(30)
    assert len(ue) == 1 and ue[0]["id"] == "e1" and ue[0]["start"].tzinfo, ue  # all-day e2 skipped

    cards, text = drive_search("แผน")
    assert cards[0]["url"] == "https://d/f1" and '"id":"f1"' in text
    drive_search("a'b\\c")
    assert seen["q"] == "fullText contains 'a\\'b\\\\c' and trashed=false", seen["q"]
    assert drive_search("none") == ([], "ไม่พบไฟล์ใน Drive ที่ตรงกับคำค้น")
    cards, text = drive_read("doc1")
    assert "เนื้อเอกสาร" in text and cards[0]["source"] == "drive"
    assert set(TOOLS) == set(STATUS) == {s["function"]["name"] for s in TOOL_SPECS}
    print("google_tools self-test OK")
