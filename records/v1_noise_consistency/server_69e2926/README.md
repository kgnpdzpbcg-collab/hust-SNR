# V1服务器运行文件索引

运行代码提交：`69e2926ebd09f0a9b8fe8dfdebfd15aa5a7a5481`。
运行目录：`/home/wang/projects/会议论文/hust/experiments/v1_noise_consistency/69e2926ebd09/`。

- [完整数值表](results/SERVER_RESULTS.md)
- [逐种子CSV](results/PER_SEED_RESULTS.csv)
- [分支原汇总JSON](results/SUMMARY.json)
- [独立指标重算及配对差值](results/INDEPENDENT_RECOMPUTATION.json)
- [独立重载核验](results/VERIFICATION.json)
- [九个最佳权重哈希及选模结果](results/FROZEN_CHECKPOINTS.json)
- [前置检查](results/PREFLIGHT.json)
- [原环境记录](results/RUNTIME.json)
- [服务器运行及核验脚本](RUN_SERVER_V1.py)
- [导出文件字节哈希](RESULTS_EXPORT_MANIFEST.json)

导出清单中的路径相对于本目录。所有逐周期预测、训练历史及日志均保留；模型权重保存在服务器，未发布。
结果仅对应固定88条Dev及本提交；347条Final-Holdout未评估。
