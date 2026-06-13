# Redrob AI Candidate Ranking System — Docker Build
# ==================================================
# CPU-only, <16GB RAM, <5 min runtime target.

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ranker/ ranker/
COPY rank.py .

# Copy pre-computed artifacts (built during precompute phase)
COPY artifacts/ artifacts/

# Default command: run ranking pipeline
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/output/submission.csv", "--artifacts", "artifacts"]
