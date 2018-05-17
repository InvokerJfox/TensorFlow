import datetime as dt
import numpy as np
from forecast.functions.datetime_operator import datetime_to_time as dtt


def smooth_data_avg(original_values, float_size=1):
    """
    对多维度数据采用均分的方法(向下取整)平滑数据；
    :param original_values: array or list
    :param float_size:
    :return:
    """
    try:
        array_data = np.asarray(original_values)
        if len(array_data.shape) > 1:
            origin_dimension_size = array_data.shape[1]  # 数据维度
        else:
            origin_dimension_size = 1

        smooth_data_size = array_data.shape[0]  # 数据
        array_data = array_data.reshape((smooth_data_size, origin_dimension_size))
        smooth_values = []  # 多维度平滑结果
    except IndexError:
        raise ValueError('original_values error')

    for dim in range(origin_dimension_size):  # 输入数据的维度
        origin_vector_values = array_data[:, dim]
        smooth_vector_values = np.zeros(smooth_data_size)

        # 2.平滑所有数值.当前后超出有效范围时，仅在有效区间内进行平均
        for i in range(smooth_data_size):
            start_index = max(0, i - float_size)
            end_index = min(smooth_data_size, i + float_size)
            avg = origin_vector_values[i] / (end_index + 1 - start_index)
            # 将值平均分给区间值
            for j in range(start_index, end_index):
                smooth_vector_values[j] += avg

        smooth_values.append(smooth_vector_values)

    return np.asarray(smooth_values)


def extend_data_dimension(origin_values, extend_backward=0, extend_forward=0):
    """
    基于前后扩展的方法扩展数据
    :param origin_values: 只会对其 [extend_backward,len - extend_forward]的数据进行扩展并返回
    :param extend_backward:
    :param extend_forward:
    :return:
    """
    try:
        origin_array = np.asarray(origin_values)

        if len(origin_array.shape) > 1:
            origin_dimension_size = origin_array.shape[0]  # 数据维度
            origin_data_size = origin_array.shape[1]  # 原始数据规模
            extend_data_size = origin_array.shape[1] - extend_backward - extend_forward  # 最终数据规模
        else:
            origin_dimension_size = 1
            origin_data_size = origin_array.shape[0]
            extend_data_size = origin_array.shape[0] - extend_backward - extend_forward

        extend_dimension_size = extend_backward + extend_forward + 1  # 扩展维度规模
        expansion_values = np.zeros((extend_data_size, origin_dimension_size * extend_dimension_size), dtype=np.float32)

    except IndexError:
        raise ValueError('original_values')

    for current_day in range(extend_data_size):  # 对所有日期进行扩展
        # 垂直扩展
        for dim, dim_value in enumerate(origin_values):  # 遍历原始数据每个维度
            for extend_dimension in range(extend_backward + 1 + extend_forward):  # 数据起始&目标偏移量
                # print('%r,%r,%r,%r' % (current_day, dim, extend_dimension_size, extend_day))
                expansion_values[current_day, dim * extend_dimension_size + extend_dimension] = dim_value[
                    current_day + extend_dimension]

    return expansion_values, extend_backward, origin_data_size - extend_forward


def fill_date_index_data_with_zero(start_date, end_date, original_date, original_values):
    """
    补全基于(排序后的)日期作为索引的数据，截断所需数据，当数据缺失时以补零法补全
    :param start_date:
    :param end_date:
    :param original_date:
    :param original_values:
    :return:
    """
    try:
        filled_data_size = (end_date - start_date).days + 1  # 最终数据的规模
        filled_values = []  # 用补0法补全
        filled_date = []
    except IndexError:
        raise ValueError('original_values')

    # 1.补全所有日期
    current_date = start_date  # 重置，用于记录
    for i in range(filled_data_size):
        filled_date.append(current_date)
        current_date += dt.timedelta(1)

    # 2.补全数据
    origin_data_index = 0  # 原始数据下标
    for current_date in filled_date:  # 遍历每一日
        while dtt(current_date) > original_date[origin_data_index]:  # 冗余数据-跳过：
            origin_data_index += 1
        else:  # dtt(current_date) <= original_date[origin_data_index]:
            if dtt(current_date) == original_date[origin_data_index]:  # 真实数据匹配
                filled_values.append(original_values[origin_data_index])
                origin_data_index += 1  # 真实数据下标下移
            else:  # 缺失数据-以0补全
                filled_values.append(0)
                origin_data_index -= 1

    return filled_date, filled_values


def fill_date_index_data_backward(start_date, end_date, original_date, original_values, backward_size):
    """
    基于回溯方法，补全数据
    :param start_date: included
    :param end_date: included
    :param original_date:
    :param original_values:
    :param backward_size:
    :return:
    """
    try:
        filled_data_size = (end_date - start_date).days + 1  # 最终数据的规模
        filled_values = []  # 用补回溯法补全
        filled_date = []
    except IndexError:
        raise ValueError('original_values')

    # 1.补全所有日期
    current_date = start_date  # 重置，用于记录
    for i in range(filled_data_size):
        filled_date.append(current_date)
        current_date += dt.timedelta(1)

    # 2.补全数据
    origin_data_index = 0  # 原始数据下标
    for current_date in filled_date:  # 遍历每一日
        while dtt(current_date) > original_date[origin_data_index]:  # 冗余数据-跳过：
            origin_data_index += 1
        else:  # dtt(current_date) <= original_date[origin_data_index]:
            if dtt(current_date) == original_date[origin_data_index]:  # 真实数据匹配
                filled_values.append(original_values[origin_data_index])
                origin_data_index += 1  # 真实数据下标下移
            else:  # 真实数据缺失，若过去一日存在数据，则填充数据；否则报错
                backward_count = 1
                # 向前n天
                while backward_count <= origin_data_index and (original_date[origin_data_index - backward_count] != dtt(
                            current_date - dt.timedelta(1))) and backward_count <= backward_size:
                    backward_count += 1
                else:
                    if backward_count <= origin_data_index and (
                                original_date[origin_data_index - backward_count] == dtt(
                                    current_date - dt.timedelta(1))):
                        filled_values.append(original_values[origin_data_index - backward_count])
                    else:  # 补全失败
                        raise ValueError('cannot filling missing values,date:%r' % current_date)

    return filled_date, filled_values


def logistic(x, a=-1, b=2, c=1, d=0):
    """
      y = a + b * 1./(1+ exp(-c * x + d)));
    :param x:
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    y = a + b / (1 + np.e ** (-1 * c * x + d))
    return y


def anti_logistic(y, a=-1, b=2, c=1, d=0):
    """
    y = args(1) + args(2) * 1./(1+ exp(-(args(3) * x + args(4))));
    :param y:
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    x = - (np.log(b / (y - a) - 1) + d) / c
    return x
