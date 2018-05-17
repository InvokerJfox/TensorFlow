import tensorflow as tf
import forecast.preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
from forecast.functions.data_functions import smooth_data_avg

# 1.读取销量、天气数据
data = pp.load_and_sort_data()


def data_division_clean_fn():
    pass

def weather_sales_features_input_fn(weathers, weathers_date, sales, sales_date):
    features = []
    labels = []

    return features, labels

def train_input_fn():
  return weather_sales_features_input_fn(df_train)

def eval_input_fn():
  return weather_sales_features_input_fn(df_forecast)