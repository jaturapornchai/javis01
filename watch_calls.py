"""Waits for a finished call in voice_agent.log, prints a short digest of it, and exits (ลุงจืด 2026-11-07
"ทุก 1 นาที แบบประหยัด token ให้มาอ่าน log ที่เปลี่ยนไป"): it reads only what was added since its last digest,
once a minute, and wakes the reader only when someone has talked and the line has then been quiet for 2 minutes.
Run it in the background; its exit is the signal. `--once` prints what is pending now and exits at once."""
import os
import re
import sys
import time

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_agent.log")
STATE = os.path.join(os.path.dirname(LOG), "sessions", "watch_offset")  # sessions/ is gitignored
KEEP = re.compile(r" (USER|BOT|TOOLS|cut |not said|another voice|echo|DeepSeek|judge failed|Traceback)|^\w+(Error|Exception)\b")
TALK = re.compile(r" (USER|BOT) ")
STAMP = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d),\d+ ")
QUIET_S, POLL_S, MAX_LINES, MAX_CHARS = 120, 60, 150, 220  # ponytail: a longer call shows its last 150 lines only


def offset():
    try:
        with open(STATE) as f:
            return int(f.read() or 0)
    except (OSError, ValueError):  # first run: start at the end, the old log was read by hand
        return os.path.getsize(LOG) if os.path.exists(LOG) else 0


def new_lines(start):
    """Whole lines added since `start` and where they end; a log trimmed by hand starts over from 0."""
    if not os.path.exists(LOG):
        return [], 0
    if os.path.getsize(LOG) < start:
        start = 0
    with open(LOG, "rb") as f:
        f.seek(start)
        data = f.read()
    whole = data[: data.rfind(b"\n") + 1]
    return whole.decode("utf-8", "replace").splitlines(), start + len(whole)


def digest(lines):
    keep = [STAMP.sub(r"\1 ", ln)[:MAX_CHARS] for ln in lines if KEEP.search(ln)]
    cut = len(keep) - MAX_LINES
    return ([f"(ตัด {cut} บรรทัดแรกออก)"] if cut > 0 else []) + keep[-MAX_LINES:]


def main():
    start = offset()
    while True:
        lines, end = new_lines(start)
        talk = [ln for ln in lines if TALK.search(ln)]
        if "--once" in sys.argv or (talk and quiet_for(talk[-1]) >= QUIET_S):
            print("\n".join(digest(lines)) or "(ไม่มีสายใหม่)")
            with open(STATE, "w") as f:
                f.write(str(end))
            return
        time.sleep(POLL_S)


def quiet_for(line):
    """Seconds since this log line was written (the log's own local clock)."""
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
    return time.time() - time.mktime(time.strptime(m[1], "%Y-%m-%d %H:%M:%S")) if m else 0


if __name__ == "__main__":
    main()
