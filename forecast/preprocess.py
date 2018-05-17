"""
Loading and preprocessing data

"""
import forecast.entry.config as config
from forecast.entry.ForecastData import ForecastData as Data
from forecast.functions.file_operator import load_str_data as lsd
from forecast.functions.datetime_operator import ymd_to_date as ymd
from forecast.functions.data_functions import fill_date_index_data_backward
from forecast.functions.data_functions import fill_date_index_data_with_zero
from forecast.functions.data_functions import smooth_data_avg
from forecast.functions.data_functions import extend_data_dimension
from forecast.functions.datetime_operator import time_to_datetime as t2d
import datetime as dt


# main data process
def load_and_sort_data():
    data = Data(config)

    # forecast sales

    # load data
    dataset_sales = lsd(config.DATASET_SALES_PATH)
    dataset_weather = lsd(config.DATASET_WEATHER_PATH)

    # sort data
    sorted_sales = ymd(dataset_sales, [0])
    sorted_sales.sort(key=lambda x: x[0])
    sorted_weather = ymd(dataset_weather, [0])
    sorted_weather.sort(key=lambda x: x[0])

    # add data
    data.set_sorted_sales(sorted_sales)
    data.set_sorted_weather(sorted_weather)

    if not data.is_data_verified():
        raise RuntimeError('data etl failed')

    return data


def preprocess_data(data):
    # 1.训练数据
    train_start_date = data.config.TRAINING_START_DAY
    train_end_date = data.config.TRAINING_END_DAY

    # 1.1.天气数据
    train_weather_date = data.dataset_train_input_weather_date
    train_weather = data.dataset_train_input_weather

    # 1.1.1.将原始数据中 选取有用的数据，当缺失数据，用补回溯法补全值
    train_weather_filled_date, train_weather_filled_values = \
        fill_date_index_data_backward(train_start_date, train_end_date, train_weather_date, train_weather,
                                      data.config.FILL_WEATHER_BACKWARD)

    # 1.1.2.对所有维度的值进行平滑处理
    train_weather_smooth_values = smooth_data_avg(train_weather_filled_values,
                                                  data.config.SMOOTH_WEATHER_DAYS)

    # 1.1.3.归纳垂直时间(即，多日)范围内的数据
    train_input_weather_extend_values, train_input_weather_start_index, train_input_weather_end_index = \
        extend_data_dimension(train_weather_smooth_values, data.config.EXTEND_WEATHER_BACKWARD_DAYS,
                              data.config.EXTEND_WEATHER_FORWARD_DAYS)
    train_input_weather_extend_date = train_weather_filled_date[
                                      train_input_weather_start_index:train_input_weather_end_index]

    # 1.2.销量数据
    train_sales_date = data.dataset_train_input_sales_date
    train_sales = data.dataset_train_input_sales

    # 1.2.1.将原始数据中 选取有用的数据，当缺失数据，用补回溯法补全值
    train_sales_filled_date, train_sales_filled_values = \
        fill_date_index_data_with_zero(train_start_date, train_end_date, train_sales_date, train_sales)

    # 1.2.2.对所有维度的值进行平滑处理
    train_sales_smooth_values = smooth_data_avg(train_sales_filled_values, data.config.SMOOTH_SALES_DAYS)

    # 1.2.3.归纳垂直时间(即，多日)范围内的数据
    train_input_sales_extend_values, train_input_sales_start_index, train_input_sales_end_index = \
        extend_data_dimension(train_sales_smooth_values, data.config.EXTEND_SALES_BACKWARD_DAYS, 0)
    train_input_sales_extend_date = train_sales_filled_date[train_input_sales_start_index:train_input_sales_end_index]

    # 1.3.对销量数据补全并平滑销量
    train_output_sales_date = data.dataset_train_output_sales_date
    train_output_sales = data.dataset_train_output_sales

    # 1.3.1.将原始数据中 选取有用的数据，当缺失数据，用补回溯法补全值
    train_output_sales_filled_date, train_output_sales_filled_values = \
        fill_date_index_data_with_zero(train_start_date, train_end_date, train_output_sales_date, train_output_sales)

    # 1.3.2.对所有维度的值进行平滑处理
    train_output_sales = smooth_data_avg(train_output_sales_filled_values, data.config.SMOOTH_SALES_DAYS)

    # 1.4.由于其他维度的数据缺失，因此将各维度(包括输出)进行数据裁剪
    max_start_index = max(train_input_weather_start_index, train_input_sales_start_index)
    min_end_index = min(train_input_weather_end_index, train_input_sales_end_index)
    size = [len(train_input_weather_extend_date), len(train_input_sales_extend_date),
            len(train_output_sales_filled_date)]

    start_index = [max_start_index - train_input_weather_start_index,
                   max_start_index - train_input_sales_start_index, max_start_index]
    end_index = [size[0] - train_input_weather_end_index + min_end_index,
                 size[1] - train_input_sales_end_index + min_end_index, min_end_index]

    data.dataset_train_input_weather_date = train_input_weather_extend_date[start_index[0]:end_index[0]]
    data.dataset_train_input_weather = train_input_weather_extend_values[start_index[0]:end_index[0]]
    data.dataset_train_input_sales_date = train_input_sales_extend_date[start_index[1]:end_index[1]]
    data.dataset_train_input_sales = train_input_sales_extend_values[start_index[1]:end_index[1]]
    data.dataset_train_output_sales_date = train_output_sales_filled_date[start_index[2]:end_index[2]]
    data.dataset_train_output_sales = train_output_sales.reshape((size[2], 1))[start_index[2]:end_index[2]]
    data.train_size = end_index[0] - start_index[0]

    # 2.预测数据
    # forecast_start_date = data.config.FORECAST_START_DAY
    # forecast_end_date = data.config.FORECAST_END_DAY

    # 2.1.天气数据
    forecast_weather_date = data.dataset_forecast_weather_date
    forecast_weather = data.dataset_forecast_weather

    # 2.1.1.将原始数据中 选取有用的数据，当缺失数据，用补回溯法补全值
    forecast_weather_filled_date, forecast_weather_filled_values = \
        fill_date_index_data_backward(t2d(forecast_weather_date[0]), t2d(forecast_weather_date[-1]),
                                      forecast_weather_date,
                                      forecast_weather, data.config.FILL_WEATHER_BACKWARD)

    # 2.1.2.对所有维度的值进行平滑处理
    forecast_weather_smooth_values = smooth_data_avg(forecast_weather_filled_values, data.config.SMOOTH_WEATHER_DAYS)

    # 2.1.3.归纳垂直时间(即，多日)范围内的数据
    data.dataset_forecast_weather, forecast_input_weather_start_index, forecast_input_weather_end_index = \
        extend_data_dimension(forecast_weather_smooth_values, data.config.EXTEND_WEATHER_BACKWARD_DAYS,
                              data.config.EXTEND_WEATHER_FORWARD_DAYS)
    data.dataset_forecast_weather_date = forecast_weather_filled_date[
                                         forecast_input_weather_start_index:forecast_input_weather_end_index]

    # 2.2.销量数据
    forecast_by_actual_sales_date = data.dataset_forecast_by_actual_sales_date
    forecast_by_actual_sales = data.dataset_forecast_by_actual_sales

    # 2.2.1.将原始数据中 选取有用的数据，当缺失数据，用补回溯法补全值
    forecast_sales_filled_date, forecast_sales_filled_values = \
        fill_date_index_data_with_zero(t2d(forecast_by_actual_sales_date[0]), t2d(forecast_by_actual_sales_date[-1]),
                                       forecast_by_actual_sales_date, forecast_by_actual_sales)

    # 2.2.2.对所有维度的值进行平滑处理
    forecast_sales_smooth_values = smooth_data_avg(forecast_sales_filled_values, data.config.SMOOTH_SALES_DAYS)

    # 2.2.3.归纳垂直时间(即，多日)范围内的数据
    data.dataset_forecast_by_actual_sales, forecast_input_sales_start_index, forecast_input_sales_end_index = \
        extend_data_dimension(forecast_sales_smooth_values, data.config.EXTEND_SALES_BACKWARD_DAYS, 0)
    data.dataset_forecast_by_actual_sales_date = forecast_sales_filled_date[
                                                 forecast_input_sales_start_index:forecast_input_sales_end_index]

    # 基于预测的销量初始数据
    data.dataset_forecast_by_forecast_sales_date = forecast_sales_filled_date[
                                                   0:data.config.EXTEND_SALES_BACKWARD_DAYS + 1]
    data.dataset_forecast_by_forecast_sales = forecast_sales_filled_values[0:data.config.EXTEND_SALES_BACKWARD_DAYS + 1]

    # 3.实际数据进行填充处理
    actual_sales_date = data.dataset_actual_sales_date
    actual_sales = data.dataset_actual_sales
    data.dataset_actual_sales_date, data.dataset_actual_sales = fill_date_index_data_with_zero(
        t2d(actual_sales_date[0]), t2d(actual_sales_date[-1]), actual_sales_date, actual_sales)

    # 数据归一化 (暂不做，看效果)
    pass

    return data
