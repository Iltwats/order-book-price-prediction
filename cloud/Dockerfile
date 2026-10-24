FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /workspace
COPY requirements.txt requirements-train.txt ./
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-cache-dir -r requirements-train.txt
COPY connectome ./connectome
COPY utils.py ./utils.py
COPY cloud/Dockerfile ./cloud/Dockerfile
ENTRYPOINT ["python", "-m", "connectome.gpu_training"]
