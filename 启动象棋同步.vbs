Option Explicit

Dim shell, desktop, files, scriptDir, logDir, runtimeFile, launcher, serviceLog
Dim url, command, attempt

Set shell = CreateObject("WScript.Shell")
Set desktop = CreateObject("Shell.Application")
Set files = CreateObject("Scripting.FileSystemObject")
scriptDir = files.GetParentFolderName(WScript.ScriptFullName)
logDir = files.BuildPath(scriptDir, "logs")
runtimeFile = files.BuildPath(logDir, "runtime.json")
launcher = files.BuildPath(scriptDir, "start_windows.cmd")
serviceLog = files.BuildPath(logDir, "windows_service.log")

If Not files.FolderExists(logDir) Then files.CreateFolder(logDir)

url = ReadRuntimeUrl(runtimeFile)
If url <> "" Then
    If ServiceReady(url) Then
        SwitchToLive url
        OpenDashboard url
        WScript.Quit 0
    End If
End If

command = Quote(launcher) & " --idle-timeout 20"
shell.Run command, 0, False

For attempt = 1 To 1200
    WScript.Sleep 250
    url = ReadRuntimeUrl(runtimeFile)
    If url <> "" Then
        If ServiceReady(url) Then
            SwitchToLive url
            OpenDashboard url
            WScript.Quit 0
        End If
    End If
Next

MsgBox "The Xiangqi service did not start. See:" & vbCrLf & serviceLog, 16, "Xiangqi"
WScript.Quit 1

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function

Sub SwitchToLive(baseUrl)
    Dim request, exitCode
    request = "curl.exe --silent --fail --output NUL --max-time 2 --request POST " & _
        "--header " & Quote("Content-Type: application/json") & " --data " & Quote("{}") & _
        " " & Quote(baseUrl & "/api/ai/follow")
    exitCode = shell.Run(request, 0, True)
End Sub

Sub OpenDashboard(baseUrl)
    Dim progId, browserCommand, browserPath, browserName, browserArgs, endQuote, firstSpace
    browserPath = ""
    On Error Resume Next
    progId = shell.RegRead("HKCU\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice\ProgId")
    browserCommand = shell.RegRead("HKCR\" & progId & "\shell\open\command\")
    If Err.Number = 0 Then
        If Left(browserCommand, 1) = Chr(34) Then
            endQuote = InStr(2, browserCommand, Chr(34))
            If endQuote > 2 Then browserPath = Mid(browserCommand, 2, endQuote - 2)
        Else
            firstSpace = InStr(1, browserCommand, " ")
            If firstSpace > 1 Then
                browserPath = Left(browserCommand, firstSpace - 1)
            Else
                browserPath = browserCommand
            End If
        End If
    End If
    Err.Clear
    If browserPath <> "" And files.FileExists(browserPath) Then
        browserName = LCase(files.GetFileName(browserPath))
        browserArgs = Quote(baseUrl & "/")
        If browserName = "chrome.exe" Or browserName = "msedge.exe" Or browserName = "brave.exe" Then
            browserArgs = "--new-window " & browserArgs
        ElseIf browserName = "firefox.exe" Then
            browserArgs = "-new-window " & browserArgs
        End If
        shell.Run Quote(browserPath) & " " & browserArgs, 1, False
        If Err.Number = 0 Then
            On Error GoTo 0
            Exit Sub
        End If
    End If
    Err.Clear
    desktop.ShellExecute baseUrl & "/", "", "", "open", 1
    On Error GoTo 0
End Sub

Function ReadRuntimeUrl(path)
    Dim stream, content, expression, matches
    ReadRuntimeUrl = ""
    If Not files.FileExists(path) Then Exit Function
    On Error Resume Next
    ' runtime.json is UTF-8/ASCII JSON.  Reading it as UTF-16 makes the URL
    ' regular expression fail even though the service is already running.
    Set stream = files.OpenTextFile(path, 1, False, 0)
    content = stream.ReadAll
    stream.Close
    If Err.Number <> 0 Then
        Err.Clear
        On Error GoTo 0
        Exit Function
    End If
    On Error GoTo 0
    Set expression = New RegExp
    expression.Pattern = """http_url""\s*:\s*""([^""]+)"""
    expression.IgnoreCase = True
    Set matches = expression.Execute(content)
    If matches.Count > 0 Then ReadRuntimeUrl = matches(0).SubMatches(0)
End Function

Function ServiceReady(baseUrl)
    Dim probe, exitCode
    ServiceReady = False
    probe = "curl.exe --silent --fail --output NUL --max-time 1 " & Quote(baseUrl & "/status")
    exitCode = shell.Run(probe, 0, True)
    ServiceReady = (exitCode = 0)
End Function
