import json
import time
import requests

# 算法任务请求地址
url = "http://192.168.40.120/cms/line/getTasking"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

# 查看当前时间
current_time = time.strftime("%H:%M:%S",time.localtime())

# 列表
vehicle_fornight_list = {} # 机动车道-夜间停车
nonvehicle_list = {}  # 非机动车道
vehicle_allday_list = {}  # 机动车道-全天停车
cameras_list = {}


# 获取任务信息
response = requests.get(url,headers=headers)

# 检查响应
if response.status_code == 200:
    # print(f"请求成功， 状态码：{response.status_code} 原因：{response.reason}")

    # 获取响应文本
    response_txt = response.text
    # print("=== 查看接口响应 ===",response_txt)

    data_json = json.loads(response_txt)
    # 遍历数据项
    i=0 # i表示有几条任务
    for one_data in data_json["data"]:
         json_item = json.dumps(one_data,ensure_ascii=False) # json.dumps()默认会把非ASCII字符转换为Unicode转义序列，例如\uXXXX
         rtsp_list = data_json["data"][i]["url"]
         cameraIndexCode = data_json["data"][i]["hk"]["cameraIndexCode"]
         nonvehicle = data_json["data"][i]["parkInfo"]["nonvehicle"]["coordinate"]
         vehicle_allday = data_json["data"][i]["parkInfo"]["vehicle_allday"]["coordinate"]
         vehicle_fornight = data_json["data"][i]["parkInfo"]["vehicle_fornight"]["coordinate"]

         
         # 提取摄像头编码
        #  camera_code = camera_list.split(":")[1]

         # 存储摄像头编码和URL到字典中
        #  cameras[camera_code] = camera_url
         
        
        #  print("查看当前所有下发任务数据：",json_item)
        #  print("-" * 200)
         i+= 1
        #  print('输出第' + str(i) + '个摄像头编码:', cameraIndexCode +"\n")
         cameras_list[i] = cameraIndexCode
        #  print("+++ 查看摄像头列表 +++",cameras_list)
        #  print("+++ 查看违停点位信息 +++",nonvehicle_list)
        #  print('查看第' + str(i) + '个违停区域:',nonvehicle)
        #  print("+++ 查看RTSP摄像头 +++",rtsp_list)
        #  print('查看第' + str(i) + '个RTSP地址:',rtsp_list)



        ####################################################
        ## 获取违停点位数据
        ####################################################

         ############# 存储非机动车道点位信息 ##############
         for index, nonvehicle_coordinates in enumerate(nonvehicle,start=1):
             coordinate_list = [(coord['x'],coord['y']) for coord in nonvehicle_coordinates]
             nonvehicle_list[i] = coordinate_list
        #  print(coordinate_list)
         
        #  print("-------------- 非机动车道点位 ---------------",nonvehicle_list)
        #  print("\n")

         # 存储 ## 机动车道-夜间停车 ## 点位信息
         for index, vehicle_fornight_coordinates in enumerate(vehicle_fornight,start=1):
             coordinate_list = [(coord['x'],coord['y']) for coord in vehicle_fornight_coordinates]
             vehicle_fornight_list[i] = coordinate_list
        #  print(coordinate_list)
        #  print("------------- 机动车道-夜间停车点位 --------------",vehicle_fornight_list)
        #  print("\n")

         # 存储 ## 机动车道-全天停车 ## 点位信息
         for index, vehicle_allday_coordinates in enumerate(vehicle_allday,start=1):
             coordinate_list = [(coord['x'],coord['y']) for coord in vehicle_allday_coordinates]
             vehicle_allday_list[i] = coordinate_list
        #  print(coordinate_list)
        #  print("\n")
        #  print("------------- 机动车道-全天停车点位 --------------",vehicle_allday_list)

    print("--------------   非机动车道点位  ---------------",nonvehicle_list,"\n")
    print("------------- 机动车道-夜间停车点位 --------------",vehicle_fornight_list,"\n")
    print("------------- 机动车道-全天停车点位 --------------",vehicle_allday_list)

        

else:
    if response.status_code == 500:
        print(f"请求失败， 状态码：{response.status_code} 原因：{response.reason}")
        print(response.text)
        pass

