# PCAP训练/测试数据划分
## 数据集预处理过程
- 只保留IPv6_TCP协议的数据
- 划分训练/测试数据
- 特征选择：去掉所有值都相等的特征列，['Ethernet_type', 'IP_version', 'IPV6_nh', 'TCP_reserved', 'TCP_urgptr']
## 数据文件
- input_csvs: csv纯文本数据
- input_npys: numpy二进制数据
- trainP: 训练数据正例，2092528例
- trainU: 训练数据无标签例，6565例
- testP: 测试数据正例，2000例
- testN: 测试数据负例，749例