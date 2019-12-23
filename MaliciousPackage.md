# 工控网络异常流量检测
2019年秋《密码学》课程项目——工控网络异常流量检测，PU Learing算法部分。
## 运行指南
以Linux/MacOS为例，Windows下部分命令会有差别
### 数据准备
指定一个目录`$DATA_ROOT`，在`$DATA_ROOT`下新建目录`input_csvs`, 
```bash
mkdir input_csvs
```
把4份csv数据文件（`testN.csv`, `testP.csv`, `trainP.csv`, `trainU.csv`）放到路径`$DATA_ROOT/input_csvs/`下
### 安装环境
项目运行所需的Python版本为Python 3.6.5, 需要pip包管理工具，并配置环境变量使得Python和pip在命令行下可以使用。  
```bash
pip install -r requirements.txt
```
运行上述命令安装`requirements.txt`中所列的依赖包。
### 运行代码
训练和测试的逻辑封装在`run_pcap.py`中，默认5次实验取平均值。`run_pcap.py`的运行方式为
```bash
python run_pcap.py --model $MODEL_NAME $DATA_ROOT
```
其中`$MODEL_NAM`是模型名称，有两个可选的值，分别为`lgbm`和`lgbm_weights`，两种模型的基学习器均为梯度提升决策树(GBDT)的lightGBM实现，前者是调整分类阈值的PU Learning方法，后者是调整样本权重的PU Learning方法。`--model`参数是可选参数，如不指定，则采用默认值`lgbm`.  
`$DATA_ROOT`是前文“数据准备”中提及的安放csv文件目录的数据根目录。该参数无默认值，必须由用户指定。  
```bash
python run_pcap.py -h
```
上述命令可以打印`run_pcap.py`的参数说明。  
运行`run_pcap.py`会将`pd`, `pf`, `auc`, `p`, `F`这5种评价指标以字典方式打印到标准输出。5种评价指标的意义和计算公式如下：  
- pd = recall = TP / (TP + FN)
- pf = FP / (FP + TN)
- auc
- p = precision = TP / (TP + FP)
- F = F-score = 2 * precision * recall / (precision + recall)