"""
configuration of static args of this package

"""
import datetime as dt

# 参数-全局
# 学习周期内，由于维度扩展原因，仅有一部分数据是最终有效的；
TRAINING_START_DAY = dt.datetime(2013, 3, 27)  # 学习数据的开始日期
TRAINING_END_DAY = dt.datetime(2015, 3, 5)  # 学习数据的停止日期 #343
# 预测周期内，由于维度扩展原因，数据实际需要的起止时间要大于预测时间
FORECAST_START_DAY = dt.datetime(2015, 4, 6)  # 预测数据的开始日期
FORECAST_END_DAY = dt.datetime(2016, 2, 6)  # 预测数据的结束日期 #61
# TRAINING_DATA_PERCENT = 0.75  # 测试-实验百分比,注:这里假定天气预报始终是准确的

# 数据路径
DATASET_SALES_PATH = 'data/024.txt'
DATASET_WEATHER_PATH = 'data/HZWeather.csv'

# 参数-数据补全
# MISSING_REFER_BOUND = 1
# MAX_MISSING_DAYS = 1

# 参与-数据补全
FILL_WEATHER_BACKWARD = 1  # 天气回溯补全

# 参数-平滑模型
SMOOTH_SALES_DAYS = 10  # 销量平滑日期量
SMOOTH_WEATHER_DAYS = 10  # 天气平滑日期量

# 参数-相邻数据扩展
EXTEND_SALES_BACKWARD_DAYS = 10  # 销量过往参考日期量
EXTEND_WEATHER_BACKWARD_DAYS = 30  # 天气过往参考日期量
EXTEND_WEATHER_FORWARD_DAYS = 10  # 天气未来参考日期量

# 参数-归一化
SALES_LOGISTIC_ARGS = [-1, 2, 0.03, 0.01]  # 销量logistic回归参数
WEATHER_LOGISTIC_ARGS = [0, 1, 0.1, 0]  # 天气影响logistic回归参数
