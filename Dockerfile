#############################################
# TimesFM 项目可用 Dockerfile (CPU 版)
# 特性:
# 1. 锁定 requirements 中版本，保证复现
# 2. 使用多阶段构建减少最终镜像体积
# 3. 支持国内源（可通过 build-arg 控制）
# 4. 统一 PYTHONPATH 与非交互后端
#############################################

# 使用官方 Python 3.10 精简镜像，兼容本地构建/运行
# 增加可切换基础镜像参数，便于使用国内镜像源
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
  PIP_NO_CACHE_DIR=1 \
  MPLBACKEND=Agg \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

# 基础系统依赖 (包含构建需要与 matplotlib 运行必要库)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  libfreetype6 \
  libpng16-16 \
  tzdata && \
  rm -rf /var/lib/apt/lists/*

########################
# deps stage (cache)
########################
FROM base AS deps
ARG USE_CHINA_MIRROR
COPY requirements.txt ./requirements.txt
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
  fi && \
  pip install --upgrade pip && \
  pip install -r requirements.txt

########################
# runtime stage
########################
FROM base AS runtime
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 复制虚拟环境 (直接使用系统 site-packages 方式，不额外建 venv)
COPY --from=deps /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# 复制项目代码
COPY . .

# 可选：若模型文件较大并会频繁更新，可在运行时 volume 挂载；这里先复制
COPY ./local_timesfm_model ./local_timesfm_model

# 设置 PYTHONPATH
ENV PYTHONPATH="/app/src"

# 健康检查脚本 (简单版)
RUN echo 'import importlib, json;import sys;mods=["numpy","pandas","torch","huggingface_hub"];print(json.dumps({m:bool(importlib.util.find_spec(m)) for m in mods}))' > /app/healthcheck.py

HEALTHCHECK --interval=1m --timeout=10s --retries=3 CMD python healthcheck.py || exit 1

# 缺省命令：打印帮助
CMD ["python", "-c", "print('Container ready. Example: python examples/fc_timesfm_score.py')"]