import os
import re
from typing import Set


def get_downloaded_pages(directory: str) -> Set[int]:
    """ 获取已下载的页码集合 """
    pattern = re.compile(r'page(\d+)_.*\.csv')
    pages = set()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            pages.add(int(match.group(1)))
    return pages


def find_missing_pages(downloaded_pages: Set[int]) -> Set[int]:
    """ 找出缺失的页码集合 """
    if not downloaded_pages:
        return set()
    max_page = max(downloaded_pages)
    all_pages = set(range(1, max_page + 1))
    return all_pages - downloaded_pages

def safe_lookup(df, condition_col, condition_val, target_col):
    """ 以某一列为条件, 查找该列值等于某值的行, 返回该行另一列的结果 """
    result = df[df[condition_col] == condition_val][target_col]
    if len(result) > 0:
        return result.values[0]
    else:
        return None