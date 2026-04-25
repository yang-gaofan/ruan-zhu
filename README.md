基于冻结 CLIP 与动量对比学习的图像表征训练与检索系统 V1.0

本项目是一套冻结 CLIP 与 MoCo 队列训练结合的图像表征学习系统，包含公开数据集训练、模型检查点输出、图片样本管理、语义检索、人工校正、表征实验记录和结果导出。

主要入口：

1. Web 系统：`python app.py`
2. 展示图片训练：`python -m deep_learning.train_moco --config configs/clip_moco_showcase_train.json`
3. CIFAR-10 训练：`python -m deep_learning.train_moco --config configs/clip_moco_cifar10_train.json --no-download`
4. 零样本评估：`python -m deep_learning.evaluate_zero_shot --config configs/clip_zero_shot_cifar10.json`
5. 当前 MoCo 检查点：`outputs/clip_moco_showcase_train/moco_v2_last.pt`
6. 评估结果：`outputs/zero_shot_cifar10`
