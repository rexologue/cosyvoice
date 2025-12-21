FROM nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY triton_trtllm_bistream/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY triton_trtllm_bistream /workspace/triton_trtllm_bistream

COPY triton_trtllm_bistream/runtime/docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
