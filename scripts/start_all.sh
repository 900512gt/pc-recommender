#!/usr/bin/env bash
# 同時啟動 RAG 聊天服務（port 8001）與 GA 推薦服務（port 8000）
#
# 用法：
#   ./scripts/start_all.sh
#
# 按 Ctrl+C 會一併關閉兩個服務。

set -euo pipefail

RAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GA_DIR="$HOME/Desktop/ga_test/code/ga_pc_builder"

if [ ! -d "$GA_DIR" ]; then
    echo "找不到 GA 專案目錄：$GA_DIR" >&2
    exit 1
fi

PIDS=()

cleanup() {
    echo -e "\n關閉服務中..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[RAG] 啟動中（http://localhost:8001）..."
(cd "$RAG_DIR" && python src/rag/server.py) &
PIDS+=($!)

echo "[GA]  啟動中（http://localhost:8000）..."
(cd "$GA_DIR" && uvicorn api:app --reload --port 8000) &
PIDS+=($!)

wait
