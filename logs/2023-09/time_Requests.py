# 市局违停项目因数据返回节点时requests超时导致程序终止
import requests

#### 超时异常捕获 ####
# try:
#     r = requests.get('http://192.168.40.120',timeout=5)

# except requests.exceptions.ConnectTimeout:
#     print('requests timeout!')

#### 捕获异常 ####


# response = requests.get('https://www.baidu.com/s?wd=完美代码')

k = {'wd':'完美代码'}
k1 = {'k1':"v1",'k2':"v2"}
response = response = requests.get('https://www.baidu.com/s',params=k1)
print(response.status_code)  # status_code 是状态码
print(response.url)

##########################
#
# 获取服务器返回的原始数据流
#
##########################

