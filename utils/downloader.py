import time
import json
import random
import requests
import pandas as pd
from tqdm import tqdm
import concurrent.futures


class sz_gov_downloader:
    def __init__(self, app_key, headers, timeout=60, save_directory='./data/', max_retries=3, max_workers=5, logger=None):
        self.app_key = app_key
        self.headers = headers
        self.timeout = timeout
        self.rows_per_page = 10000
        self.save_directory = save_directory
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.logger = logger

    def get_url(self, page):
        return f"https://opendata.sz.gov.cn/api/29200_00903509/1/service.xhtml?page={page}&rows={self.rows_per_page}&appKey={self.app_key}"
    
    def fetch_weather_data(self, url, page):
        """ 获取天气数据 """
        time.sleep(random.uniform(0.1, 0.5)) # 添加随机延迟以避免请求过于频繁
        try:
            # 使用POST请求，参数通过URL传递
            response = requests.post(url, headers=self.headers, timeout=self.timeout)
            print(response.json())
            if response.status_code == 200:
                try:
                    # 成功请求
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.error(f"页面 {page} 响应不是有效的JSON格式")
                        self.logger.error(f"原始响应: {response.text[:200]}...")
                    else:
                        print(f"页面 {page} 响应不是有效的JSON格式")
                        print(f"原始响应: {response.text[:200]}...")
                    return None
            else:
                if self.logger:
                    self.logger.error(f"页面 {page} 请求失败，状态码: {response.status_code}")
                    self.logger.error(f"响应内容: {response.text[:200]}...")
                else:
                    print(f"页面 {page} 请求失败，状态码: {response.status_code}")
                    print(f"响应内容: {response.text[:200]}...")
                return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"页面 {page} 请求异常: {e}")
            else:
                print(f"页面 {page} 请求异常: {e}")
            return None

    def save_to_csv_pandas(self, data, page):
        """ 保存数据到CSV文件 """
        if not data:
            if self.logger:
                self.logger.warning("没有数据可保存")
            else:
                print("没有数据可保存")
            return
        
        if 'data' not in data or not data['data']:
            if self.logger:
                self.logger.warning("数据格式不符合预期")
            else:
                print("数据格式不符合预期")
            return
        
        try:
            # 使用pandas创建DataFrame并保存为CSV
            df = pd.DataFrame(data['data'])
            # filename
            time_span = '{}to{}'.format(df['DDATETIME'].min().replace(' ','_'), df['DDATETIME'].max().replace(' ','_'))
            time_span = time_span.replace('-','').replace(':','')
            filename = f"page{page}_rows{self.rows_per_page}_{time_span}.csv"
            # save
            df.to_csv(self.save_directory+filename, index=False, encoding='utf-8-sig')
            if self.logger:
                self.logger.info(f"页面 {page} 数据已保存到 {self.save_directory+filename}")
            else:
                print(f"页面 {page} 数据已保存到 {self.save_directory+filename}")
            return filename

        except Exception as e:
            if self.logger:
                self.logger.error(f"使用pandas保存CSV文件时出错: {e}")
            else:
                print(f"使用pandas保存CSV文件时出错: {e}")
            return None

    def download_page_with_retry(self, page):
        """ 下载单个页面，包含重试机制 """
        url = self.get_url(page)
        flag_success = False
        for attempt in range(self.max_retries):
            try:
                weather_data = self.fetch_weather_data(url, page)
                if weather_data:
                    csv_file = self.save_to_csv_pandas(weather_data, page)
                    if csv_file:
                        flag_success = True
                    break
                else:
                    if self.logger:
                        self.logger.warning(f"页面 {page} 获取数据失败")
                    else:
                        print(f"页面 {page} 获取数据失败")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"页面 {page} 处理过程中发生错误: {e}")
                else:
                    print(f"页面 {page} 处理过程中发生错误: {e}")
            # 指数退避策略
            wait_time = 2 ** attempt  
            if self.logger:
                self.logger.info(f"页面 {page} 第 {attempt+1} 次尝试失败，等待 {wait_time} 秒后重试...")
            else:
                print(f"页面 {page} 第 {attempt+1} 次尝试失败，等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)

        if not flag_success:
            if self.logger:
                self.logger.error(f"页面 {page} 经过 {self.max_retries} 次尝试后仍然失败")
            else:
                print(f"页面 {page} 经过 {attempt+1} 次尝试后仍然失败")
    
    def download_pages_concurrently(self, pages_list: list):
        """ 使用线程池并发下载 """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        # 提交所有任务
            future_to_page = {executor.submit(self.download_page_with_retry, page): page for page in pages_list}
            # 使用tqdm创建进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_page), 
                            total=len(future_to_page), 
                            desc='下载进度'):
                page = future_to_page[future]
                try:
                    future.result()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"页面 {page} 生成异常: {e}")
                    else:
                        print(f"页面 {page} 生成异常: {e}")