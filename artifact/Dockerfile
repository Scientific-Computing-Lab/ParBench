# ParBench — Reproducibility Artifact (NeurIPS 2026)
# Purpose: Reproduce all tables (T1-T5) and figures (F2-F7, C.1-C.4) from raw data.
# No GPU, API keys, or special hardware required.
#
# Build:  docker build -t parbench .
# Run:    docker run --rm -v $(pwd)/output:/app/output parbench ./reproduce.sh

FROM python:3.12-slim

# System dependencies for libclang
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libclang-dev \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (lock file for exact reproducibility)
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy project source
COPY pyproject.toml .
COPY harness/ harness/
COPY c_augmentation/ c_augmentation/
COPY scripts/ scripts/
COPY schema/ schema/
COPY specs/ specs/
COPY manifest.jsonl .
COPY config/ config/

# Bundle evaluation data and pre-computed analysis summaries
COPY results/ results/

# Fix paths.json for container (host paths won't exist)
RUN python3 -c "import json; p='/app'; json.dump({'project_root':p,'downloads_root':p,'hecbench_root':p}, open('config/paths.json','w'), indent=4)"

# Install project in editable mode (registers harness and c_augmentation packages)
RUN pip install --no-cache-dir -e .

# Copy and prepare reproduce script
COPY reproduce.sh .
RUN chmod +x reproduce.sh

CMD ["echo", "Usage: docker run --rm -v $(pwd)/output:/app/output parbench ./reproduce.sh"]
