import json
import csv
import pandas as pd


class FileUtil(object):
    """
    文件工具类
    """
    @classmethod
    def read_csv_data(cls, data_path):
        """
        从csv文件中读取数据
        :param data_path:
        :return:
        """
        df = pd.read_csv(data_path, header=None, low_memory=False)
        return df

    @classmethod
    def read_excel_data(cls, data_path):
        """
        从excel文件中读取数据
        :param data_path:
        :return:
        """
        df = pd.read_excel(data_path)
        return df

    @classmethod
    def write_csv_data(cls, data_path, columns_name, data):
        """
        将数据写入csv文件
        :param data_path:
        :return:
        """
        with open(data_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns_name)
            writer.writerows(data)


    @classmethod
    def json_to_csv(cls, read_path, write_path):
        """
        将json文件转为csv文件
        :param data_path:
        :return:
        """
        with open(read_path, 'r', encoding='utf-8') as data:
            dataset = json.load(data)
            for i in range(len(dataset)):
                for j in range(len(dataset[0])):
                    with open(write_path, 'a+', newline="") as f:
                        csv_write = csv.writer(f)
                        try:
                            csv_head = dataset[i][j]
                        except IndexError:
                            continue
                        try:
                            csv_write.writerow(csv_head)
                        except UnicodeEncodeError:
                            continue



