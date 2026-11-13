' Starts the secretary with no window at logon (Task Scheduler "javis voice agent", ลุงจืด 2026-11-07 "ตั้งเปิดอัตโนมัติ"),
' and again 30 s after it stops: at logon Ollama or the network may not be up yet, and warm-up then ends the program.
' ponytail: 5 starts at most, so a second copy (preview, test_call.py) cannot reload the models forever;
' rotates voice_agent.log to voice_agent.old.log once it passes 10 MB
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
logPath = sh.CurrentDirectory & "\voice_agent.log"
oldPath = sh.CurrentDirectory & "\voice_agent.old.log"
For i = 1 To 5
    If fso.FileExists(logPath) Then
        If fso.GetFile(logPath).Size > 10485760 Then
            On Error Resume Next
            If fso.FileExists(oldPath) Then fso.DeleteFile oldPath, True
            fso.MoveFile logPath, oldPath
            On Error GoTo 0
        End If
    End If
    sh.Run "cmd /c .venv\Scripts\python.exe -X utf8 -W ignore voice_agent.py >> voice_agent.log 2>&1", 0, True
    WScript.Sleep 30000
Next
