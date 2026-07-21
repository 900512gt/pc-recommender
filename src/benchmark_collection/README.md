先列出 unique chip 清單並正規化命名
從 PassMark 抓這 52 筆跑分（手動或半自動都可以，量不大）
建 chip_benchmark.json 對照表
寫 join script 把跑分廣播到 ga_database 的 272 個項目，順便加上 benchmark_source = "chip_mapped" 欄位
檢查有沒有 chip 找不到，那少數幾個再用插值/回歸補