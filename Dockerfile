# RepoSage container image.
#
# Design choice: the vector index (chroma_db/) and the target repo
# (target_repos/httpx) are built INTO the image, not generated on first
# request. The app never writes to either at runtime -- it only reads
# from them -- so baking them in means every deployed container is
# byte-for-byte reproducible, and there's no first-request indexing
# latency or need for a persistent volume. Trade-off: this makes the
# image larger and the build slower (embedding model download + full git
# clone happen at build time), which is the right trade for a read-only
# workload like this one.

FROM python:3.11-slim

# git is needed at RUNTIME, not just build time -- risk_scoring.py calls
# `git log` via subprocess on every risk_assessment request.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Full (non-shallow) clone -- required for risk_scoring's commit-history
# queries. A shallow clone would make every file report ~1 commit,
# breaking risk classification entirely (this bit us for real in Step 1).
RUN git clone https://github.com/encode/httpx.git target_repos/httpx

COPY app/ app/

# Build the vector index now, not at first request. No GOOGLE_API_KEY
# needed for this step -- only the embedding model (downloaded here,
# needs network access during the build), not the LLM.
RUN python -m app.indexing.vector_store

EXPOSE 8000

# GOOGLE_API_KEY (and optionally the LANGCHAIN_* tracing vars) must be
# injected at RUNTIME by whatever platform runs this image -- via its
# secrets/env-var mechanism, never baked into the image or committed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]