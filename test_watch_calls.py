"""python test_watch_calls.py: the log watcher on a made-up log (no server, no models)."""
import os
import tempfile
import time

import watch_calls as w

d = tempfile.mkdtemp()
w.LOG, w.STATE = os.path.join(d, "voice_agent.log"), os.path.join(d, "watch_offset")
old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 300))
with open(w.LOG, "w", encoding="utf-8") as f:
    f.write(f"{old},1 - public tunnel closed, redialling in 10 s\n"
            f"{old},2 ab12 USER (0.00s): สวัสดีครับ\n"
            f"{old},3 ab12 BOT  (1.0s | boss 1 judge 0 voice 0): สวัสดีครับ\n"
            f"{old},4 ab12 Connection(2) Could not bind to x - [WinError 10049]\n"
            f"{old},5 ab12 cut a panel line (judge): x\n"
            "ValueError: boom\n"
            f"{old},6 ab12 BOT  half a li")  # still being written: not read yet

lines, end = w.new_lines(0)
assert len(lines) == 6 and end < os.path.getsize(w.LOG), "a line still being written waits for its newline"
got = w.digest(lines)
assert [g.split(" ", 2)[-1][:4] for g in got[:3]] == ["USER", "BOT ", "cut "], got  # tunnel and bind noise left out
assert got[0].startswith(old[11:] + " ab12 "), "date and milliseconds dropped, call tag kept"
assert got[-1] == "ValueError: boom"
assert 290 < w.quiet_for(lines[1]) < 400
assert w.new_lines(10**9)[0] == lines, "a log trimmed by hand is read again from the start"
print("ok")
