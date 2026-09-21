FROM python:3.10-slim

# Two environments keep the pinned inference stack independent of the current UI.
RUN python -m venv /opt/ui && python -m venv /opt/inference
COPY hf_demo/ui-requirements.txt /tmp/ui-requirements.txt
RUN /opt/ui/bin/pip install --no-cache-dir -r /tmp/ui-requirements.txt
RUN /opt/inference/bin/pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY pyproject.toml README.md LICENSE vgs_decoding.py vgs_llavamed_vqarad.py vgs_medgemma_vqarad.py /app/
RUN /opt/inference/bin/pip install --no-cache-dir . && /opt/inference/bin/pip check
COPY hf_demo/app.py hf_demo/worker.py /app/hf_demo/
RUN useradd --create-home --uid 1000 demo && mkdir -p /home/demo/.cache/huggingface /home/demo/.cache/gradio && chown -R demo:demo /home/demo/.cache && chmod 700 /home/demo/.cache/huggingface /home/demo/.cache/gradio
USER demo
ENV VGS_INFERENCE_PYTHON=/opt/inference/bin/python \
    HF_HOME=/home/demo/.cache/huggingface \
    GRADIO_TEMP_DIR=/home/demo/.cache/gradio \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PORT=7860
EXPOSE 7860
CMD ["/opt/ui/bin/python", "/app/hf_demo/app.py"]
