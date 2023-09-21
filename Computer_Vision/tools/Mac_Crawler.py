from cgitb import html
import os
from unittest.mock import patch
from urllib import response
from wsgiref import headers
import requests
import re
import time
from tqdm import tqdm
from progress.bar import Bar  # 进度条库

"""请求网址"""
""" MAC实验室网站图片爬取 """

headers = {
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://mac.xmu.edu.cn/members.htm',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'

}
response = requests.get('https://mac.xmu.edu.cn/members.htm', headers=headers)  # 获取网页解析后的源代码
html = response.text

# 解析网页 正则表达式去匹配
urls = re.findall('src="(.*?)" ', html)
# dir_name = 'D:\\_Project_\\VsCode\\python\\pic\\'
save_dir_name = 'D:\\Computer_Vision\\Books\\mac\\'  # 图片保存路径
if not os.path.exists(save_dir_name):
    os.mkdir(save_dir_name)

# 打印匹配结果(可注销)
# --print(urls)

## 进度条设置
bar = Bar("图片下载中", fill='#', max=100, suffix='%(percent)d%%')

print('图片数量统计中..')
# 循环数组取出图片
count = 0
x = 0
for url in urls:
    # --print(url) # 打印是否输出数组元素(可注销)

    # 进行数组元素拼接成URL
    h = 'https://mac.xmu.edu.cn/'  # mac.xmu.edu.cn后面要加上'/' 否则会转移 https://mac.xmu.edu.cn\img/members/wumingrui.jpg 导致路径错误
    p = os.path.join(h, url)  # 拼接方法

    # --print(p) # 打印是否拼接(可注销)

    # time.sleep(0.1)
    time.sleep(0.01)
    file_name = url.split('/')[-1]
    response = requests.get(p, headers=headers)

    # print('输出p---------：' + p)
    with open(save_dir_name + '/' + file_name, 'wb') as f:
        f.write(response.content)
        # print('图片下载完成:'+ p)

    ## 单张图片进度加载
    count += 1
    # for n in bar.iter(range(count)):
    #     time.sleep(0.01)
## count 统计图片数量
# print('+'*100,count)
for n in bar.iter(range(count)):
    time.sleep(0.001)




