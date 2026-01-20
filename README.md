# 项目简介
单细胞药物反应预测与训练管线。当前运行配置 = `config.py` 基础 + `overrides/run_global.json` 覆盖：使用双通路特征（ssGSEA + PROGENy）、药物机制指纹、ProtoMechanism 模型，只学习全局 F1 网格阈值。

## 当前配置要点
- 模型：`proto_mech`，`proto_num=13`，`proto_topk=4`，`proto_mech_lambda=1.3`，`proto_diversity_lambda=0.12`。
- 通路特征：`pathway_methods=["ssgsea","progeny"]`；资源映射  
  - ssGSEA：`data/resource/c2.cp.reactome.v2025.1.Hs.symbols.gmt`  
  - PROGENy：`data/resource/progeny_human_top500.tsv`  
  其他参数：`pathway_min_genes=5`，`pathway_max_genes=500`，`pathway_rank_alpha=0.75`。
- 药物指纹：开启，方法 `["progeny"]`，静态/动态权重 0.55/0.45，缓存 `data/resources/drug_fingerprint_cache.npz`。
- 机制 FiLM：开启（隐藏层 48），让药物机制调制通路通道。
- 阈值：`threshold_strategy="f1_grid"`，`threshold_beta=1.1`，仅一个全局阈值。
- 输入/输出路径：`DATA_ROOT`、`RESULTS_ROOT`、`LOG_ROOT` 见 `config.py`；输入 `data/processed/*.h5ad`，输出位于 `${RESULTS_ROOT}/<run-name>/`。

## 方法亮点
- 双通路表征：同时计算 ssGSEA（Reactome GMT）和 PROGENy（扰动权重 TS V），覆盖代谢/信号通路与扰动响应。
- 药物机制指纹：PROGENy 机制向量（静态+动态权重）直接拼入模型，使药物先验约束特征学习。
- ProtoMechanism：原型稀疏路由（top-k）、容量/多样性正则、机制 FiLM，使原型对药物机制敏感并避免塌缩。
- 决策：直接用全局 F1 网格阈值，评估使用 95% t 分布区间。

## 数据流与训练流程
1. 数据加载：从 `data/processed/*.h5ad` 读取，按 `DATASET_METADATA` 解析标签与正负值。
2. 特征构建：`features/pipeline.py` 依次计算 ssGSEA + PROGENy 通路分数，拼接基因表达；`features/drug_fingerprints.py` 加载 PROGENy 指纹，生成静态/动态机制先验。
3. 模型训练：`models/factory.py` 构建 ProtoMechanism；`code/run.py` 训练，原型稀疏选择 + 容量/多样性约束 + 机制 FiLM。
4. 阈值学习：验证集 F1 网格搜索得到单一全局阈值。
5. 推理与保存：应用全局阈值得到标签与指标（ACC/F1/AUROC/AUPRC）；写出模型权重与各类中间产物。

## 产物清单（${RESULTS_ROOT}/<run-name>/）
- `results.csv`：各数据集指标汇总。
- `summary.json`：阈值与指标汇总。
- `prototype_diagnostics.json`：原型使用率、容量、路由分布。
- `model_checkpoint.npz`：模型参数。

## 运行示例
```bash
# 使用 config.py + run_global.json 覆盖运行
PIPELINE_OVERRIDES="$(cat overrides/run_global.json)" \
python code/run.py --config config.py --run-name proto_mech_global
```

## 目录速览
```
config.py                # 路径/数据集元信息 & PIPELINE_FLAGS
overrides/run_global.json # 本次运行的覆盖项（模型/通路/指纹/阈值）
code/
  run.py                 # 训练与推理主流程
  data_loader.py         # 数据读取与预处理
  features/              # 通路/指纹特征构建
  models/                # ProtoMechanism
  postprocess/           # 阈值选择
scripts/                 # 指纹缓存构建工具
data/                    # processed 输入、results 输出、resource 资源
```
