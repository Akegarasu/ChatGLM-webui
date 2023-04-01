@echo off
cd /D "%~dp0"
if exist .venv goto :start

echo "Setup VENV"
python -m venv .venv
call .venv\Scripts\activate.bat

echo "Install dependencies"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install --upgrade -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
goto :run

:start
echo "Start VENV"
call .venv\Scripts\activate.bat
goto :run

:run
echo "Start WebUI"
python webui.py
pause
