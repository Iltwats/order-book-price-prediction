FROM python:3.11-slim-bookworm

# Build dependencies are installed only while creating the image. The checks
# themselves run with --network none and cannot install anything.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/connectome-requirements.txt
RUN python -m pip install --no-cache-dir \
    -r /tmp/connectome-requirements.txt onnx==1.17.0 pytest==8.3.5

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace
CMD ["python", "scripts/run_linux_checks.py", "--inside"]
