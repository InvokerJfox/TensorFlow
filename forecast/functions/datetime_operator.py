import time
import datetime as dt


def ymd_to_date(data_list, date_column=None):
    """
    convert str list to time list;
    :param date_column:
    :param data_list:
    :return:
    """
    if date_column is None:
        date_column = [0]
    processed_data = []
    rows = iter(data_list)
    for row in rows:
        try:
            new_row_data = []
            for column, value in enumerate(row):
                if column in date_column:
                    new_row_data.append(time.strptime(row[column], "%Y/%m/%d"))
                else:
                    new_row_data.append(row[column])
            processed_data.append(new_row_data)
        except:
            raise ValueError('datetime value is exception')
    return processed_data


def datetime_to_time(value):
    f = "%Y/%m/%d %H:%M:%S"
    return time.strptime(value.strftime(f), f)


def float_to_datetime(value):
    """

    :param value:
    :return:
    """
    return dt.datetime.fromtimestamp(value)


def time_to_datetime(value):
    """

    :param value:
    :return:
    """
    return dt.datetime.fromtimestamp(time_to_float(value))


def time_to_float(value):
    return time.mktime(value)


def float_to_time(value):
    return time.gmtime(value)


def datetime_to_str(value):
    f = "%Y/%m/%d %H:%M:%S"
    return value.strftime(f)
