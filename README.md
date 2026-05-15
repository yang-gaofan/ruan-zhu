基于冻结 CLIP 与动量对比学习的图像表征训练与检索系统 V1.0

本项目是一套冻结 CLIP 与 MoCo 队列训练结合的图像表征学习系统，包含公开数据集训练、模型检查点输出、图片样本管理、语义检索、人工校正、表征实验记录和结果导出。

主要入口：

1. Web 系统：`python app.py`
2. MoCo 预训练：`python -m deep_learning.training.train_moco --config configs/final_clip_moco_pretrain_4060.json`
3. 最终分类评估：`python -m deep_learning.evaluation.linear_probe --config configs/final_clip_moco_linear_probe_4060.json --no-download`
4. 当前 MoCo 检查点：`outputs/final_clip_moco_pretrain/moco_v2_last.pt`
5. 评估结果：`outputs/final_clip_moco_linear_probe`
