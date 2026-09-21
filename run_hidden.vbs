' Start the YuE2 webUI with no window at all, then open it in the browser.
' Logs: logs\server.log     Stop it with: stop.cmd
Option Explicit
Dim shell, fso, root
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
shell.Run "cmd /c serve.cmd", 0, False
WScript.Sleep 8000
shell.Run "http://127.0.0.1:7860", 1, False
