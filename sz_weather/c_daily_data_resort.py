from utils.data_processor import daily_data_processor


if __name__ == "__main__":
    data_dir = "./storage/sz_weather/data_raw/"
    save_directory = "./storage/sz_weather/daily_data_raw/"
    max_workers = 16    # 并发任务数
    max_grid_id = 4231  # 最大 GRIDID 值
    processor = daily_data_processor(
        data_dir=data_dir,
        max_workers=max_workers,
        save_dir=save_directory,
        max_grid_id=max_grid_id
    )
    processor.run()