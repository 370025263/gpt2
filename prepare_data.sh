# !/bin/bash
# 修正：原来这里把 HuggingFace token 明文写进了脚本，而仓库是公开的，任何人都能拿去用你的账号。
# token 已经留在 git 历史里，改这一行删不掉，必须去 https://huggingface.co/settings/tokens 把旧 token 作废。
# 以后从环境变量读：export HF_TOKEN=...；fineweb 是公开数据集，不登录也能下载，没有 token 时跳过登录。
pip install -r requirements.txt && { [ -z "$HF_TOKEN" ] || huggingface-cli login --token "$HF_TOKEN"; } && python down_data.py
