# ChatGLM-webui

A webui for ChatGLM made by THUDM. [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)


## Install

### requirements

python3.10

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade -r requirements.txt
```

or

```shell
bash install.sh
```

## Run

```shell
python webui.py
```

### Args

`--model-path`: specify model path. If this parameter is not specified manually, the default value is `THUDM/chatglm-6b`. Transformers will automatically download model from huggingface.

`--listen`: launch gradio with 0.0.0.0 as server name, allowing to respond to network requests

`--port`: webui port

`--share`: use gradio to share

`--precision`: fp16, int4, int8

`--cpu`: use cpu