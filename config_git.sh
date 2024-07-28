#!/bin/bash
git remote set-url origin git@github.com:370025263/gpt2.git

# 配置 Git 用户信息
git config --global user.email "370025263@qq.com"
git config --global user.name "370025263"

# 提示用户输入私钥内容
echo "请输入您的 SSH 私钥内容（输入完成后按 Ctrl+D 结束）："
private_key=$(cat)

# 创建临时文件存储私钥
temp_key_file=$(mktemp)
echo "临时私钥文件路径: $temp_key_file"

# 将私钥内容写入临时文件并设置权限
echo "$private_key" > "$temp_key_file"
chmod 400 "$temp_key_file"

# 启动 ssh-agent 并添加私钥
eval "$(ssh-agent -s)"
ssh-add "$temp_key_file"

# 测试连接到 GitHub
ssh -T git@github.com


