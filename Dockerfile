# RAG API（server.py，port 8001）
# 只打包這支服務實際會 import 到的東西：src/rag/ + data/rag_chunks.jsonl。
# 資料管線（crawler / filter / database / benchmark_collection）不會被跑進這個 image。
#
# Build（context 必須是本 repo 根目錄）：
#   docker build -t rag-api .
# Run：
#   docker run -p 8001:8001 -e OPENAI_API_KEY=sk-xxx rag-api

FROM python:3.11-slim

WORKDIR /app

COPY src/rag/requirements.txt src/rag/requirements.txt
RUN pip install --no-cache-dir -r src/rag/requirements.txt

COPY src/rag src/rag
COPY data/rag_chunks.jsonl data/rag_chunks.jsonl

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["python", "src/rag/server.py"]
