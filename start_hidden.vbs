' Starts the secretary with no window at logon (Task Scheduler "javis voice agent", ลุงจืด 2026-11-07 "ตั้งเปิดอัตโนมัติ"),
' and again 30 s after it stops: at logon Ollama or the network may not be up yet, and warm-up then ends the program.
' ponytail: 5 starts at most, so a second copy (preview, test_call.py) cannot reload the models forever;
' voice_agent.log only grows (gitignored), trim it by hand
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
For i = 1 To 5
    sh.Run "cmd /c .venv\Scripts\python.exe -X utf8 -W ignore voice_agent.py >> voice_agent.log 2>&1", 0, True
    WScript.Sleep 30000
Next
