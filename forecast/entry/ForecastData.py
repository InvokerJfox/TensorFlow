import datetime as dt

import numpy as np

from forecast.functions.datetime_operator import datetime_to_time as d2t


class ForecastData:
    """

    """
    # user only get preprocessed data
    config = None
    DATA_TO_BE_PROCESSED_SIZE = 2
    data_preprocessed_size = 0

    # date and sales & weather mapping with index
    dataset_origin_sales_date = []  # type='float'
    dataset_origin_sales = []
    dataset_origin_weather_date = []  # type='float'
    dataset_origin_weather = []

    dataset_train_input_weather_date = []  # type='float'
    dataset_train_input_weather = []
    dataset_train_input_sales_date = []  # type='float'
    dataset_train_input_sales = []
    dataset_train_output_sales_date = []  # type='float'
    dataset_train_output_sales = []

    dataset_actual_sales_date = []  # type='float'
    dataset_actual_sales = []
    dataset_actual_weather_date = []  # type='float'
    dataset_actual_weather = []

    dataset_forecast_by_actual_sales_date = []  # 基于当前之前若干天的<真实销量>进行销量维度扩展
    dataset_forecast_by_actual_sales = []
    dataset_forecast_by_forecast_sales_date = []  # 基于当前天之前若干天的<预测销量>进行销量维度扩展
    dataset_forecast_by_forecast_sales = []
    dataset_forecast_weather_date = []  # type='float'
    dataset_forecast_weather = []

    # data start & end
    dataset_sales_start_date = dt.datetime
    dataset_sales_end_date = dt.datetime
    dataset_weather_start_date = dt.datetime
    dataset_weather_end_date = dt.datetime

    # data size
    train_size = 0
    forecast_size = 0

    def __init__(self, conf):
        # load config
        self.config = conf

        self.train_size = (conf.TRAINING_END_DAY - conf.TRAINING_START_DAY).days - \
                          max(conf.EXTEND_SALES_BACKWARD_DAYS,
                              conf.EXTEND_WEATHER_BACKWARD_DAYS) - conf.EXTEND_WEATHER_FORWARD_DAYS
        self.forecast_size = (conf.FORECAST_END_DAY - conf.FORECAST_START_DAY).days

        # verified request
        if self.train_size < 0:
            raise ValueError('TRAINING_END_DAY < TRAINING_START_DAY',
                             conf.TRAINING_END_DAY, conf.TRAINING_START_DAY)
        elif self.forecast_size < 0:
            raise ValueError('FORECAST_END_DAY < FORECAST_START_DAY', conf.FORECAST_END_DAY,
                             conf.FORECAST_START_DAY)

        print('Train Data Size:', self.train_size)
        print('Forecast Data Size:', self.forecast_size)

    def set_sorted_sales(self, date_sales):
        """
        set the sales which sorted by date
        :param self:
        :param date_sales: [[date,sales],...]
        :return:
        """
        train_start_date = d2t(self.config.TRAINING_START_DAY)
        train_end_data = d2t(self.config.TRAINING_END_DAY)
        # 预测数据是在预测日期范围进行扩展得到的
        forecast_start_date = d2t(self.config.FORECAST_START_DAY)
        forecast_end_date = d2t(self.config.FORECAST_END_DAY)
        forecast_sales_start_date = d2t(
            self.config.FORECAST_START_DAY - dt.timedelta(self.config.EXTEND_SALES_BACKWARD_DAYS))
        forecast_sales_end_date = d2t(self.config.FORECAST_END_DAY)

        try:
            self.dataset_sales_start_date = start_date = date_sales[0][0]
            self.dataset_sales_end_date = end_date = date_sales[-1][0]
        except IndexError:
            raise ValueError('input date_sales error')

        # verified data
        if train_start_date < start_date or end_date < train_end_data:
            raise ValueError(
                'train_start_date < dataset_sales_start_date or train_end_data > dataset_sales_end_dat')
        if forecast_start_date < start_date or end_date < forecast_end_date:
            raise ValueError(
                'forecast_start_date < dataset_sales_start_date or forecast_end_date > dataset_sales_end_dat')

        # save data
        # origin data
        origin_date_sales = np.asarray(date_sales)
        self.dataset_origin_sales_date = origin_date_sales[:, 0]
        self.dataset_origin_sales = np.asarray(origin_date_sales[:, 1], dtype=float)

        # train data
        train_date_sales = np.asarray(
            list(x for x in date_sales if train_start_date <= x[0] <= train_end_data))
        self.dataset_train_input_sales_date = train_date_sales[:, 0]
        self.dataset_train_input_sales = np.asarray(train_date_sales[:, 1], dtype=float)
        self.dataset_train_output_sales_date = train_date_sales[:, 0]
        self.dataset_train_output_sales = np.asarray(train_date_sales[:, 1], dtype=float)

        # forecast weather & sales data
        # sales of actual
        forecast_by_actual_date_sales = np.asarray(
            list(x for x in date_sales if forecast_sales_start_date <= x[0] <= forecast_sales_end_date))
        self.dataset_forecast_by_actual_sales_date = forecast_by_actual_date_sales[:, 0]
        self.dataset_forecast_by_actual_sales = np.asarray(forecast_by_actual_date_sales[:, 1], dtype=float)

        # sales of backward sales
        forecast_by_forecast_date_sales = np.asarray(
            list(x for x in date_sales if forecast_sales_start_date <= x[0] <= forecast_start_date))
        self.dataset_forecast_by_forecast_sales_date = forecast_by_forecast_date_sales[:, 0]
        self.dataset_forecast_by_forecast_sales = np.asarray(forecast_by_forecast_date_sales[:, 1], dtype=float)

        # actual data
        actual_date_sales = np.asarray(list(x for x in date_sales if forecast_start_date <= x[0] <= forecast_end_date))
        self.dataset_actual_sales_date = actual_date_sales[:, 0]
        self.dataset_actual_sales = np.asarray(actual_date_sales[:, 1], dtype=float)

        # successful processed
        self.data_preprocessed_size += 1

    def set_sorted_weather(self, date_weather):
        """

        :param date_weather:
        :return:
        """

        train_start_date = d2t(self.config.TRAINING_START_DAY)
        train_end_data = d2t(self.config.TRAINING_END_DAY)
        forecast_start_date = d2t(self.config.FORECAST_START_DAY)
        forecast_end_date = d2t(self.config.FORECAST_END_DAY)
        forecast_weather_start_date = d2t(
            self.config.FORECAST_START_DAY - dt.timedelta(self.config.EXTEND_WEATHER_BACKWARD_DAYS))
        forecast_weather_end_date = d2t(
            self.config.FORECAST_END_DAY + dt.timedelta(self.config.EXTEND_WEATHER_FORWARD_DAYS))

        try:
            self.dataset_weather_start_date = start_date = date_weather[0][0]
            self.dataset_weather_end_date = end_date = date_weather[-1][0]
        except IndexError:
            raise ValueError('input date_sales error')

        # verified data
        # forecast day must have weather information
        if train_start_date < start_date or end_date < train_end_data:
            raise ValueError(
                'train_start_date < dataset_sales_start_date or train_end_data > dataset_sales_end_dat')
        if forecast_start_date < start_date or end_date < forecast_end_date:
            raise ValueError(
                'forecast_start_date < dataset_sales_start_date or forecast_end_date > dataset_sales_end_dat')

        # save data
        # origin data
        origin_date_weather = np.asarray(date_weather)
        self.dataset_origin_weather_date = origin_date_weather[:, 0]
        self.dataset_origin_weather = np.asarray(origin_date_weather[:, 1:3], dtype=float)  # max & min weather

        # train data
        train_date_weather = np.asarray(list(x for x in date_weather if train_start_date <= x[0] <= train_end_data))
        self.dataset_train_input_weather_date = train_date_weather[:, 0]
        self.dataset_train_input_weather = np.asarray(train_date_weather[:, 1:3], dtype=float)  # max & min weather

        # forecast & actual data
        forecast_date_weather = np.asarray(
            list(x for x in date_weather if forecast_weather_start_date <= x[0] <= forecast_weather_end_date))
        self.dataset_forecast_weather_date = forecast_date_weather[:, 0]
        self.dataset_forecast_weather = np.asarray(forecast_date_weather[:, 1:3], dtype=float)

        actual_date_weather = np.asarray(
            list(x for x in date_weather if forecast_start_date <= x[0] <= forecast_end_date))
        self.dataset_actual_weather_date = actual_date_weather[:, 0]
        self.dataset_actual_weather = np.asarray(actual_date_weather[:, 1:3], dtype=float)

        # successful processed
        self.data_preprocessed_size += 1

    def is_data_verified(self):
        """

        :return:
        """
        if self.data_preprocessed_size == self.DATA_TO_BE_PROCESSED_SIZE:
            return True
        else:
            raise False
