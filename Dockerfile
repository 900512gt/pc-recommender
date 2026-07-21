# GA API（api.py，port 8000）
# config.py 的 MATCHED_FILES 指向 repo 根目錄的 match_data/matched_part*.jsonl
# （config.py 用 parent.parent.parent 往上找），所以 build context 必須是
# ga_test/ 根目錄，不能只在 code/ga_pc_builder/ 底下 build，否則會缺資料。
#
# Build（context 必須是 ga_test/ 根目錄）：
#   docker build -t ga-api .
# Run：
#   docker run -p 8000:8000 ga-api

FROM python:3.11-slim

WORKDIR /app

COPY code/ga_pc_builder/requirements.txt code/ga_pc_builder/requirements.txt
RUN pip install --no-cache-dir -r code/ga_pc_builder/requirements.txt

COPY code/ga_pc_builder code/ga_pc_builder
COPY match_data/matched_part1.jsonl match_data/matched_part1.jsonl
COPY match_data/matched_part2.jsonl match_data/matched_part2.jsonl
COPY match_data/matched_part3.jsonl match_data/matched_part3.jsonl

WORKDIR /app/code/ga_pc_builder

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
