import logging
from utils.downloader import sz_gov_downloader


if __name__ == "__main__":
    # 下载参数
    start_page, end_page = 0, 200000 #　total_rows = 1203130430
    directory = "./storage/sz_weather/data_raw/"
    log_directory = f"./logs/download_log_{start_page}to{end_page}.log"
    max_workers = 1 # 并发任务数
    max_retries = 3 # 单次请求最大重试次数
    pages_list = list(range(start_page, end_page + 1))

    # 请求参数
    app_key = "" # 请替换为实际的 appKey
    if not app_key:
        raise ValueError("请在代码中设置 app_key 变量的值！")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "appKey": app_key
    }

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_directory),  # 输出到文件
            # logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )

    # downloader 实例
    downloader = sz_gov_downloader(
        app_key,
        headers=headers,
        timeout=120,
        save_directory=directory,
        max_retries=max_retries,
        max_workers=max_workers,
        logger=logging
        )
    
    print(f"开始页数: {start_page}, 结束页数: {end_page}, 最大并发任务数: {max_workers}, 最大重试次数: {max_retries}")
    
    # 并发下载
    downloader.download_pages_concurrently(pages_list)