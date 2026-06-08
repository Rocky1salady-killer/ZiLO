# 基于已有的 perception:vision 镜像
FROM perception:vision

# 设置工作目录
WORKDIR /workspace/zilo-main

# 拷贝你的项目代码到镜像中
COPY . .

# 如果有 requirements.txt 就取消注释安装
# RUN pip install --no-cache-dir -r requirements.txt

# 禁用 ROS 默认入口，改为 bash
CMD ["bash"]

