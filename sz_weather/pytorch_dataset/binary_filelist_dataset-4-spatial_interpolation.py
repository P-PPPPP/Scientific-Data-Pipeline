import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Literal

from src.core.utils import get_global_config, get_global_logger
from src.core.data import build_mask_generator, normalize_position, geo_to_3d_coords, Normalizer, Features


class BinaryFilelistDataset(Dataset):
    """Dataset for meteorological binary data with memory mapping support."""
    
    def __init__(
        self, 
        file_dir: str, 
        file_list: List[Path], 
        normalizer: Normalizer,
        stage: Literal["train", "test", "val"]
    ):
        """
        Meteorological Dataset for binary data files.
        
        Args:
            file_dir: Directory containing binary files
            file_list: List of binary files to include
            normalizer: Data normalization transformer
            stage: Dataset stage ('train', 'val', 'test')
        """
        self.logger = get_global_logger()
        self.config = get_global_config()
        self.bin_dir = Path(file_dir)
        self.file_list = file_list
        self.normalizer = normalizer
        self.stage = stage

        # 坐标归一化方法
        if getattr(self.config.model, 'feature_encoder', None):
            self.coords_emb_method = getattr(self.config.model.feature_encoder, 'coords_emb_method', 'max_min')
        else:
            self.coords_emb_method = 'max_min'
        
        # 初始化
        self._load_metadata()
        self._load_auxiliary_data()
        self._initialize_memory_maps()
        self._build_sequence_index()
        self.mask_generator = build_mask_generator(self.config.task.sensor_failure)

        # 日志
        self.logger.dataset_loading_info(stage, size=len(file_list))

    def _load_metadata(self) -> None:
        """Load dataset metadata from JSON file."""
        metadata_path = self.config.dataset.metadata_path
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_features = self.metadata['num_features']
        self.num_grids = self.metadata['num_grids']
        self.time_steps_per_day = self.metadata['time_steps_per_day']
        self.dtype = self.config.dtype

    def _load_auxiliary_data(self) -> None:
        # 坐标
        coords_raw = np.load(self.config.dataset.coords_data_path)
        self.coords_raw = torch.tensor(coords_raw)
        if self.coords_emb_method == 'max_min':
            self.coords_normed = normalize_position(self.coords_raw)
        elif self.coords_emb_method == 'spherical_coordinates':
            self.coords_normed = geo_to_3d_coords(self.coords_raw)
        else:
            raise NotImplementedError(f'坐标归一化方法 {self.coords_emb_method} 方法未实现')
        # 地势
        elevation_data = np.load(self.config.dataset.elevation_data_path)
        elevation_data = torch.tensor(elevation_data, dtype=self.dtype)
        elev_min = torch.min(elevation_data)
        elev_max = torch.max(elevation_data)
        self.elevation_data = (elevation_data - elev_min) / (elev_max - elev_min)
        # 日期
        date_path = self.config.dataset.date_data_path
        with open(date_path, 'r') as f:
            self.date_data = json.load(f)

    def _initialize_memory_maps(self) -> None:
        """Initialize memory maps for binary files."""
        self.memmaps = []
        self.valid_files = []
        
        for bin_file in self.file_list:
            try:
                mmap = np.memmap(
                    bin_file,
                    dtype=np.float32,
                    mode='r',
                    shape=(self.time_steps_per_day, self.num_grids, self.num_features),
                )
                self.memmaps.append(mmap)
                self.valid_files.append(bin_file)
            except Exception as e:
                self.logger.log_message('warning', f'Failed to load file {bin_file.name}: {e}')
                continue
        
        if not self.memmaps:
            raise RuntimeError("No valid binary files could be loaded")

    def _build_sequence_index(self) -> None:
        """Build index of all sequences across all files."""
        self.sequence_index = []
        
        for file_idx, bin_file in enumerate(self.valid_files):
            file_stem = bin_file.stem
            for seq_idx in range(self.time_steps_per_day):
                self.sequence_index.append({
                    'file_idx': file_idx,
                    'seq_idx': seq_idx,
                    'file_stem': file_stem,
                    'filename': bin_file.name
                })

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int) -> Features:
        if idx >= len(self.sequence_index):
            raise IndexError(f"Index {idx} out of dataset bounds")

        seq_info = self.sequence_index[idx]
        
        # Get mask for this sample
        mask = self.mask_generator()
        
        # Load and process data
        data_tensor_normed, data_tensor_masked = self._load_and_process_data(
            seq_info['seq_idx'], 
            seq_info['file_idx'], 
            mask
        )
        
        # Get coordinates and datetime
        coords = self._get_coords()
        datetime = self._get_datetime(seq_info['file_stem'], seq_info['seq_idx'])

        return Features(
            data=data_tensor_masked,
            target=data_tensor_normed,
            coords=coords,
            datetime=datetime,
            elevation=self.elevation_data,
            mask=mask
        )

    def _load_and_process_data(
        self, 
        seq_idx: int, 
        file_idx: int, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and process data for a specific sequence."""
        # Calculate start time and load data
        mmap = self.memmaps[file_idx]
        sequence_data = mmap[seq_idx].copy()
        
        # Convert to tensor and normalize
        sequence_tensor = torch.from_numpy(sequence_data).to(dtype=self.dtype)
        data_tensor_normed = self.normalizer(sequence_tensor)
        
        # Apply mask
        data_tensor_masked = self._apply_mask(data_tensor_normed, mask)
        
        return data_tensor_normed, data_tensor_masked

    def _get_coords(self) -> torch.Tensor:
        """Get normalized coordinates."""
        return self.coords_normed

    def _get_datetime(self, file_stem: str, seq_idx: int) -> torch.Tensor:
        """Get and normalize datetime information."""
        time_data = self.date_data[file_stem][seq_idx]
        
        # Parse datetime and extract components
        time_data = pd.to_datetime(time_data)
        datetime_components = [
            time_data.year, time_data.month, time_data.day, 
            time_data.hour, time_data.minute
        ]
        
        datetime_tensor = torch.tensor(datetime_components, dtype=self.dtype)
        return datetime_tensor

    def _apply_mask(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to data tensor."""
        masked_data = data.clone()
        masked_data[mask, :] = 0.0
        return masked_data

    def __del__(self):
        """Clean up memory maps on destruction."""
        for mmap in self.memmaps:
            if hasattr(mmap, '_mmap'):
                mmap._mmap.close()
        self.memmaps.clear()