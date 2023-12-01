import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Global_Wind_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, freq='h'):
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        self.flag = flag
        self.freq = freq
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        if self.flag == "train":
            self.raw_data = np.load(os.path.join(self.root_path, "train", "data_value_train.npy"), allow_pickle=True)
            self.raw_time = np.load(os.path.join(self.root_path, "train", "data_time_train.npy"), allow_pickle=True)
            length = len(self.raw_data)
            self.raw_data = self.raw_data[: int(length * 0.8)]
            self.raw_time = self.raw_time[: int(length * 0.8)]
        elif self.flag == "val":
            self.raw_data = np.load(os.path.join(self.root_path, "train", "data_value_train.npy"), allow_pickle=True)
            self.raw_time = np.load(os.path.join(self.root_path, "train", "data_time_train.npy"), allow_pickle=True)
            length = len(self.raw_data)
            self.raw_data = self.raw_data[int(length * 0.8):]
            self.raw_time = self.raw_time[int(length * 0.8):]
        else:
            self.raw_data = np.load(os.path.join(self.root_path, "test", "data_value_test_x.npy"), allow_pickle=True)
            self.raw_time = np.load(os.path.join(self.root_path, "test", "data_time_test_x.npy"), allow_pickle=True)

        raw_data = self.raw_data
        raw_time = self.raw_time
        print(self.raw_data.shape)
        print("==== " + self.flag + " data sorted load finished ====")

        data = raw_data.astype(float)

        df_stamp = raw_time
        data_stamp_all = []
        if self.flag != "test":
            df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            data_stamp_all = data_stamp
        else:
            for i in range(len(df_stamp)):
                pd_datatime = pd.to_datetime(df_stamp[i])
                data_stamp = time_features(pd.to_datetime(pd_datatime), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
                data_stamp_all.append(data_stamp)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp_all

    def __getitem__(self, index):
        if self.flag != "test":
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            seq_x = self.data_x[index]
            seq_x_mark = self.data_stamp[index]

            return seq_x, seq_x_mark

    def __len__(self):
        if self.flag != "test":
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        else:
            return len(self.data_x)