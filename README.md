# 多站点多变量气象预测 

本项目为北航2023年《机器学习》课程代码示例

## 环境
环境配置请参考：https://github.com/cure-lab/LTSF-Linear

## 数据
下载数据集存放在dataset路径下

## 训练
```
python run.py --is_training 1
```
模型文件将保存在checkpoints路径下

## 预测
```
python run.py --is_training 0
```
结果文件将保存在results路径下

## 代码
代码细节请参考论文：Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
