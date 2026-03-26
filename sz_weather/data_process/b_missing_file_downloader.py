
import logging
from utils.downloader import sz_gov_downloader
from utils.functions import get_downloaded_pages, find_missing_pages


if __name__ == "__main__":
    # 下载参数
    directory = "./storage/sz_weather/data_raw/"
    log_directory = "./logs/redownload_log.log"
    max_workers = 1 # 并发任务数
    max_retries = 3 # 单次请求最大重试次数
    timeout = 240 # 请求超时时间

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

    # 搜索缺失的页码并重新下载
    downloaded_pages = get_downloaded_pages(directory)
    missing_pages = find_missing_pages(downloaded_pages)
    missing_pages = sorted(missing_pages)

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
        timeout=timeout,
        save_directory=directory,
        max_retries=max_retries,
        max_workers=max_workers,
        logger=logging
        )
    
    # 下载缺失的页码
    if missing_pages:
        print(f"timeout: {timeout}, max_workers: {max_workers}, max_retries: {max_retries}")
        print(f"Found {len(missing_pages)} missing pages. Starting re-download...")
        print(f"Missing pages: {missing_pages}")
        downloader.download_pages_concurrently(missing_pages)
    else:
        print("No missing pages found.")