# Software Defect Detection
Course assignment for **Software Testing** & **Cryptography** 2019 Fall, School of Software, Tsinghua University
## TODO
- lgbm结果作图
- PPT
    - 基础模型：lgbm
    - method 1: threshold （重点是推导），缺点：在p/u特征差别较大时效果不明显
    - method 2: loss weight （重点是调参），缺点：需要调参
## Object
Software defect/malicious package detection by machine learning approaches
## Data
Software defect detection: `CK` & `NASA` datasets with 10%, 20%, 30% data for training and the remaining for testing.  
Malicious package detection: private real world data
## Project Structure
- `classifiers.py`: collections of classifiers
- `config.json`: configuration file
- `data_loaders.py`: methods for loading data
- `formatter.py`: dumps results to csv files
- `main.py`: project entrance
- `metrics.py`: evaluation metrics
- `processors.py`: re-samplers
