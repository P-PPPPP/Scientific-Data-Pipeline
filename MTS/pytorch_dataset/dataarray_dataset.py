import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.core.utils import get_global_config, get_global_logger
from src.core.data import Normalizer, Features


class DataArrayDataset(Dataset):
    def __init__(self, data_array_set, normalizer: Normalizer, stage):
        self.config = get_global_config()
        self.logger = get_global_logger()

        self.device = self.config.device
        self.dtype = self.config.dtype

        self.data = data_array_set['data']
        self.date = data_array_set['date']
    
        # 序列长度定义
        self.input_seq_len = self.config.task.input_seq_len 
        self.pred_seq_len = self.config.task.pred_seq_len 
        
        # 预先获取节点和通道数量，方便后续 broadcast
        self.num_nodes = self.data.shape[1]
        self.num_channels = self.data.shape[2]
        self.total_seq_len = self.data.shape[0]
        self.window_size = self.input_seq_len + self.pred_seq_len

        assert self.num_nodes == self.config.dataset.num_nodes, f"数据节点数 {self.num_nodes} 与配置节点数 {self.config.dataset.num_nodes} 不匹配"
        assert self.num_channels == self.config.dataset.num_channels, f"数据通道数 {self.num_channels} 与配置通道数 {self.config.dataset.num_channels} 不匹配"
        
        # data 转换为 torch tensor 
        data_tensor = torch.tensor(self.data, dtype=self.dtype)
        # 数据归一化
        if normalizer is not None:
            self.data_tensor = normalizer(data_tensor)
        # date 转换为 torch tensor
        dt_index = pd.DatetimeIndex(self.date)
        datetime_np = np.stack([
            dt_index.year,
            dt_index.month,
            dt_index.day,
            dt_index.hour,
            dt_index.minute
        ], axis=-1)
        self.date_tensor = torch.tensor(datetime_np, dtype=self.dtype) 

        # 日志
        self.logger.dataset_loading_info(stage, size=self.__len__())
    
    def __len__(self) -> int:
        return self.total_seq_len - self.window_size + 1

    def __getitem__(self, idx: int) -> Features:
        start_idx = idx
        mid_idx = idx + self.input_seq_len
        end_idx = idx + self.window_size
        
        # data
        data_tensor = self.data_tensor[start_idx:mid_idx]
        target_tensor = self.data_tensor[mid_idx:end_idx]
        # datetime
        datetime_tensor = self.date_tensor[start_idx:end_idx]

        return Features(
            data=data_tensor,
            target=target_tensor,
            datetime=datetime_tensor
        )