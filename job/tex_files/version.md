完成了V1.0： 实现功能：Logistic / SVM / RF / XGBoost 模型预测，和回测引擎1.0和2.0。
问题： 
**DATA SNOOPING**因子所做的处理都使用了整个数据集，如RankP, RankM, 标准化时用了全数据集方差/均值。
**NO CROSS-VALIDITION**直接用网格搜索得到最佳超参数，打乱了有序的数据做验证，并且用了未来的参数。