FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY triton_trtllm_bistream /workspace/triton_trtllm_bistream

COPY triton_trtllm_bistream/runtime/docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
