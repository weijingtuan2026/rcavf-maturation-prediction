# RCAVF成熟预测模型

本项目依据《Machine Learning with Segmental Duplex Ultrasound Predicts Ultrasound-Defined Radiocephalic Fistula Maturation Before Surgery and at the 2-Week Landmark》实现三个二分类算法：弹性网逻辑回归、XGBoost和CatBoost。每个算法均训练两个时间点模型：

- 术前 Segmental 模型：10项术前指标。
- 术后2周 Full 模型：上述10项加5项术后2周动态指标。

程序将结果追加到待预测Excel工作表右侧。每个算法追加4列：术前成熟概率、术前判断、术后2周成熟概率、术后2周判断。直接回写默认开启，并在同一文件夹生成带时间戳的备份。三个程序依次对同一个Excel运行后，文件中会保留12个模型结果列。

## 重要限制

论文和补充材料只给出了汇总统计、超参数和测试集性能，没有提供367例患者级训练数据或已训练权重。因此，本项目不能从文档恢复原论文模型。运行时必须选择一份带6周结局的患者级训练Excel；程序会使用论文固定超参数训练模型，再预测另一份Excel。当前结果只能用于研究验证，不能在外部验证和临床审批前替代医生判断。

## 安装

建议使用Python 3.10至3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

macOS上的XGBoost还需要OpenMP运行库：

```bash
brew install libomp
```

Windows激活虚拟环境：

```text
.venv\Scripts\activate
```

## 窗口方式

在项目文件夹内分别运行：

```bash
python lr_model.py
python xgboost_model.py
python catboost_model.py
```

窗口中选择：

1. 带6周结局的训练Excel和工作表。
2. 待预测Excel和工作表。
3. 训练结局列名。
4. 是否直接追加到待预测Excel。

结局列推荐使用 `maturation_failure`，其中1表示6周未成熟/失败、0表示成熟。若列名使用 `是否成熟`，则1表示成熟、0表示未成熟。也支持“成熟/未成熟”文本。

## 命令行方式

```bash
python catboost_model.py \
  --train 历史训练数据.xlsx \
  --predict 5名患者.xlsx \
  --train-sheet Sheet1 \
  --predict-sheet Sheet1 \
  --outcome maturation_failure
```

加入 `--no-direct-write --output 5名患者_预测结果.xlsx` 可保留原预测Excel不变。

## 必需字段

程序支持下列英文标准列名，也支持常用中文别名。二分类指标可用0/1、是/否或有/无。连续指标应为数值，不要在单元格中附加单位。

| 时间点 | 英文标准列名 | 中文含义 |
|---|---|---|
| 术前 | `age` | 年龄（岁） |
| 术前 | `diabetes` | 糖尿病（0/1） |
| 术前 | `albumin` | 白蛋白（g/L） |
| 术前 | `radial_artery_diameter` | 桡动脉直径（mm） |
| 术前 | `radial_artery_ri` | 术前桡动脉RI |
| 术前 | `arterial_wall_calcification` | 动脉壁钙化（0/1） |
| 术前 | `cv_diameter_distal` | 远端头静脉直径（mm） |
| 术前 | `min_cv_diameter` | 三段头静脉最小直径（mm） |
| 术前 | `segmental_diameter_variability` | 三段直径标准差/均值 |
| 术前 | `mean_segmental_depth` | 三段头静脉平均深度（mm） |
| 2周 | `access_flow_2w` | 通路流量（mL/min） |
| 2周 | `mean_diameter_change_2w` | 三段平均直径变化（2周减术前，mm） |
| 2周 | `mean_depth_change_2w` | 三段平均深度变化（2周减术前，mm） |
| 2周 | `focal_stenosis_2w` | 局灶性狭窄（0/1） |
| 2周 | `radial_artery_ri_change_2w` | 桡动脉RI变化（2周减术前） |

三个术前复合字段可以不直接提供。若同时提供远端、中段、近端的直径和深度，程序会自动计算：

- `cv_diameter_mid`、`cv_diameter_proximal`
- `cv_depth_distal`、`cv_depth_mid`、`cv_depth_proximal`

三个2周变化字段也可由配对原始值自动计算。可使用：

- `vein_diameter_distal_2w`、`vein_diameter_mid_2w`、`vein_diameter_proximal_2w`
- `vein_depth_distal_2w`、`vein_depth_mid_2w`、`vein_depth_proximal_2w`
- `radial_artery_ri_2w`

## 模型参数与判断阈值

模型使用补充材料Table S3的固定参数。判断阈值来自正文Table 5，并针对论文建模事件“未成熟”应用：

| 算法 | 术前未成熟阈值 | 术后2周未成熟阈值 |
|---|---:|---:|
| LR | 0.298 | 0.339 |
| XGBoost | 0.221 | 0.187 |
| CatBoost | 0.254 | 0.412 |

Excel中的成熟概率为 `1 - 未成熟概率`。当未成熟概率低于相应阈值时输出“成熟”，否则输出“未成熟”。

