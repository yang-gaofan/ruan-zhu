冻结 CLIP + MoCo 训练目录用于图像表征学习实验。

目录结构：

```text
data/        数据集构建与图像增强
models/      MoCo v2 与冻结 CLIP 投影模型
training/    自监督表征训练入口
evaluation/  线性分类评估与零样本评估
```

推荐训练命令：

```powershell
D:\anaconda\envs\dl\python.exe -m deep_learning.training.train_moco --config configs\final_clip_moco_pretrain_4060.json

D:\anaconda\envs\dl\python.exe -m deep_learning.evaluation.linear_probe --config configs\final_clip_moco_linear_probe_4060.json --no-download
```

最终模型路线为冻结 CLIP 视觉主干、MoCo v2 投影训练、线性分类头评估。
