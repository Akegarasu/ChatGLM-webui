@echo off

set GIT=git\\cmd\\git.exe
set PYTHON=py310\\python.exe

%PYTHON% webui_local.py --precision int4 --model-path "./model/chatglm-6b"

pause
exit /b