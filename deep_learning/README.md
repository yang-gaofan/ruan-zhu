冻结 CLIP + MoCo 训练目录用于图像表征学习实验。

推荐训练命令：

```powershell
D:\anaconda\envs\dl\python.exe -m deep_learning.train_moco --config configs\clip_moco_showcase_train.json

D:\anaconda\envs\dl\python.exe -m deep_learning.train_moco --config configs\clip_moco_cifar10_train.json --no-download
```

服务器训练可去掉 `sample-limit`，并把 `batch-size`、`queue-size`、`epochs` 调大。
