# 阶段B：fine-tune BERT 五大维度情感分类

## 环境需求（你已具备）
- PyTorch + CUDA（你的 4080 ✓）
- transformers, scikit-learn

如果缺 scikit-learn: pip install scikit-learn

## 训练资料
train_dimensions.jsonl（16,392则，五大维度标注）
放在跟 train_bert.py 同一资料夹。

## 使用顺序

### 1. 先小资料试跑（确认没错，约5-10分钟）
```
python train_bert.py --model ckip --sample 1000 --epochs 1
```
看到各维度的 Acc/F1 出现就代表脚本正常。

### 2. 正式训练中研院 BERT（约1-3小时）
```
python train_bert.py --model ckip --epochs 3
```
产出：bert_ckip_dims.pt

### 3. 训练 Google BERT（比较用）
```
python train_bert.py --model google --epochs 3
```
产出：bert_google_dims.pt

### 4. 比较两个的 F1 分数，选表现好的接进 GA

## 模型选项
- --model ckip   : 中研院 ckiplab/bert-base-chinese
- --model google : Google bert-base-chinese
- --model hfl    : 哈工大 RoBERTa（可选，通常最强）

## 参数
- --epochs 3   : 训练轮数（3够用）
- --batch 16   : 批次大小（4080可以16，显存不够降到8）
- --lr 2e-5    : 学习率（标准值，别乱改）
- --sample N   : 只用前N则（试跑用）

## 输出解读
```
效能: Acc 0.85  Macro-F1 0.78
温控: Acc 0.88  Macro-F1 0.75
...
平均 Macro-F1: 0.76
```
Macro-F1 越高越好。中研院 vs Google 比这个数字选模型。

## 下一步（阶段C）
选好模型后，用它跑全部 50,593 则评论 → 每个零件的面向分数。
