# 用于模型预测，不断以 batch 形式从 redis in_queue 队列读取收到数据，模型预测后，送入 redis result 队列
# 可以运行多个， python model_predict.py X 来运行，其中 X 表示使用的显卡号
import base64
import io
import json
import os
import sys
import datetime
import time
import yaml
import math
import pdb
import cv2
import numpy as np
import redis
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from PIL import Image

from model import ft_net_swinv2
from utils import fuse_all_conv_bn

gpu_ids = '3'
which_epoch = 99
name = 'swinv2_all_1'
linear_num = 512
ms = '1'

### Load config ###
# load the training config
config_path = os.path.join("./model", name, "opts.yaml")
with open(config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
stride = config['stride']
nclasses = 2
linear_num = config['linear_num']
str_ids = gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

print("We use the scale: %s"%ms)
str_ms = ms.split(",")
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

h, w, = 256, 128
transform = T.Compose([
    T.Resize(size=(h, w), interpolation=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()

# Load model
def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

model_structure = ft_net_swinv2(nclasses, (h, w), linear_num=linear_num)
model = load_network(model_structure)
model = model.eval()
if use_gpu:
    model = model.cuda()
model = fuse_all_conv_bn(model)
print("Model is prepared, waiting data...")

BATCH = 16
BATCH_TIME = 3
dummy_forward_input = torch.rand(BATCH, 3, h, w).cuda()
model = torch.jit.trace(model, dummy_forward_input)
print("JIT trace on model.")


global result_list
result_list = []

global redis_client
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
print("Redis client started.")
device = torch.device(f"cuda:{gpu_ids[0]}")


def save_img(img, picid, value, ans):
    date = time.strftime("%m%d", time.localtime(time.time()))
    base_path = "/home/xxxy/Documents/EBike/output_pic/" + date
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = os.path.join(base_path, ans + "_" + str(value) + "_" + str(picid) + ".png")
    print("path:", path)
    print("save_img")
    print("" * 40)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = img.transpose(1, 2, 0)[:, :, ::-1]
    cv2.imwrite(path, img)

def base2picture(base64_data):
    img = base64.b64decode(base64_data)
    img = io.BytesIO(img)
    img = Image.open(img)
    return img

def get_result_json(data_json):
    result = {
                "Analysis": {
                    "Action": {
                        "RecongnitionBasis": {
                            "Basis": 9,
                            "ImageList": [],
                            "SystemID": "35029772002011000001",
                            "SystemName": "其他",
                            "SystemType": 99,
                        },
                        "Types": "107",
                    },
                    "RecongnitionItems": {}
                },
                "Capture": {
                    "Deviation": data_json["Capture"]["Deviation"],
                    "EventType": data_json["Capture"]["EventType"],
                    "FullImage": {
                        "Data": "",
                        "FileFormat": data_json["Capture"]["FullImage"]["FileFormat"],
                        "FileHash": "",
                        "FileSize": data_json["Capture"]["FullImage"]["FileSize"],
                        "Height": data_json["Capture"]["FullImage"]["Height"],
                        "ImageID": data_json["Capture"]["FullImage"]["ImageID"],
                        "Path": "",
                        "Time": data_json["Capture"]["FullImage"]["Time"],
                        "Width": data_json["Capture"]["FullImage"]["Width"]
                    },
                    "Height": data_json["Capture"]["Height"],
                    "Latitude": data_json["Capture"]["Latitude"],
                    "Longitude": data_json["Capture"]["Longitude"],
                    "SubImageList": [
                        {
                            "Data": "",
                            "FileFormat": data_json["Capture"]["SubImageList"][0]["FileFormat"],
                            "FileHash": "",
                            "FileSize": data_json["Capture"]["SubImageList"][0]["FileSize"],
                            "ImageID": data_json["Capture"]["SubImageList"][0]["ImageID"],
                            "LeftTopX": 0,
                            "LeftTopY": 0,
                            "Path": "",
                            "RightBtmX": 0,
                            "RightBtmY": 0,
                            "Time": data_json["Capture"]["SubImageList"][0]["Time"],
                            "Type": 9
                        }
                    ]
                },
                "Policy": {
                    "CreateTime": data_json["Policy"]["CreateTime"],
                    "GridID": data_json["Policy"]["GridID"],
                    "HasFullImage": 0,
                    "InfoKind": data_json["Policy"]["InfoKind"],
                    "IsBuckleFace": 0,
                    "LogicSensorID": data_json["Policy"]["LogicSensorID"],
                    "LogicSensorType": data_json["Policy"]["LogicSensorType"],
                    "ObjectSnapshotId": data_json["Policy"]["ObjectSnapshotId"],
                    "ObjectType": data_json["Policy"]["ObjectType"],
                    "OriginSystemID": "35020000032001000225",
                    "OriginSystemName": "厦门电动自行车交通违法抓拍系统项目",
                    "RecognitionQuality": data_json["Policy"]["RecognitionQuality"],
                    "SenseTime": data_json["Policy"]["SenseTime"],
                    "SensorID": data_json["Policy"]["SensorID"],
                    "SensorType": data_json["Policy"]["SensorType"],
                    "SourceSystemID": "35020000032001000225",
                    "SourceSystemName": "厦门电动自行车交通违法抓拍系统项目",
                    "SubObjType": data_json["Policy"]["SubObjType"]
                },
                "Version": 1001
            }
    return result

def load_batch_imgs():
    num_img = 0
    saved_images_count = 0  # Counter for saved non-violating images
    MAX_NON_VIOLATING_IMAGES = 100  # Maximum number of non-violating images to save

    global result_list
    import pdb
    while True:
       try:
            n_img = 0
            datas = []

            picids = []
            num_imgs = []
            data_infos = []
            start_time = time.time()
            
            while n_img < BATCH and time.time() - start_time < BATCH_TIME:
                data = redis_client.lpop("ebike_in_xmu")
                if data is None:
                    continue
                data = json.loads(json.loads(data.decode("utf-8")))
                if data is None and data == "":
                    continue
                num_img += 1
                n_img += 1
                # picid = data["picId"]
                picid = data["Capture"]["SubImageList"][0]["ImageID"]
                # imgdata = data["picBase64"]
                imgdata = data["Capture"]["SubImageList"][0]["Data"]
                # print(data["Policy"]["ObjectSnapshotId"])
                # print(imgdata)
                img = base2picture(imgdata)
                img = np.array(img)
                r_i, g_i, b_i = cv2.split(img)
                img = cv2.merge([b_i, g_i, r_i])
                img1 = Image.fromarray(img)
                img1 = transform(img1)
                img1 = img1.unsqueeze(0)
                datas.append(img1)
                picids.append(picid)
                num_imgs.append(num_img)
                data_infos.append(data)
                
            with torch.no_grad():
                #pdb.set_trace()
                datas = torch.cat(datas)
                datas = datas.to(device)
            
                #  TODO: 根据电动车模型修改
                # import pdb
                # pdb.set_trace()
                # 加时间
                #start_detc_time = time.time()
                outputs = model(datas)
                #end_dect_time = time.time()
                #infer_time = end_dect_time - start_dect_time

                logits = outputs
                value, preds = torch.max(logits.data, 1)
                mask = (preds == 1) & (value < 1.4)
                preds[mask] = torch.tensor(0).cuda()

                
                curr_time = datetime.datetime.now()

                for i in range(datas.shape[0]):

                    if preds[i] == 0:
                        ans = "buchaobiao"
                        print(data_infos[i]["Policy"]["ObjectSnapshotId"], "不超标"," ",curr_time)
                        #save_img(datas[i].cpu().numpy(), picids[i], value[i].cpu().item, ans)
                        saved_images_count += 1
                    else:
                        ans = "chaobiao"
                        print(data_infos[i]["Policy"]["ObjectSnapshotId"], "超标 - 发送至服务器"," ",curr_time)
                    # print(data_infos[i]["Capture"]["FullImage"]["ImageID"], ans)
                        #save_img(datas[i].cpu().numpy(), picids[i], value[i].cpu().item(), ans)

                    print("+++  查看计数 ++++ :" + saved_images_count)
                    if saved_images_count >= MAX_NON_VIOLATING_IMAGES:
                        return
                    #result = {
                    #    "picId": picids[i],
                    #    "code": 200,
                    #    "message": {
                    #        "num_img": num_imgs[i],
                    #        "class": ans,
                    #        "value": str(value[i].cpu().item()),
                    #    },
                    #}
                    #pdb.set_trace()
                    if ans == "chaobiao":
                        
                        result = data_infos[i]
                        if "Analysis" not in result:
                            result["Analysis"] = {}

                        if "Action" not in result["Analysis"]:
                            result["Analysis"]["Action"] = {}

                        if "RecongnitionBasis" not in result["Analysis"]["Action"]:
                            result["Analysis"]["Action"]["RecongnitionBasis"] = {}

                        if "Types" not in result["Analysis"]["Action"]:
                            result["Analysis"]["Action"]["Types"] = ""

                        if "Describe" not in result["Analysis"]["Action"]:
                            result["Analysis"]["Action"]["Describe"] = ""

                        
                        # 修改Action里的系统id
                        result["Analysis"]["Action"]["RecongnitionBasis"]["SystemID"] = "35020000032001000228"
                        result["Analysis"]["Action"]["RecongnitionBasis"]["SystemName"] = "厦大非机动车违停检测系统平台"

                        # 如果types原来就存在，则把107加上，不能直接覆盖
                        if result["Analysis"]["Action"]["Types"]  != "":
                            result["Analysis"]["Action"]["Types"] = result["Analysis"]["Action"]["Types"] + ",107"
                        else:
                            result["Analysis"]["Action"]["Types"] = "107"
                        
                        # result["Analysis"]["Action"]["Types"] = "107"
                        # 如果Describe 之前存在，那么也添加后续
                        if result["Analysis"]["Action"]["Describe"] != "":
                            result["Analysis"]["Action"]["Describe"] = result["Analysis"]["Action"]["Describe"] + ",非机动车超标"
                        else:
                            result["Analysis"]["Action"]["Describe"] = "非机动车超标"
                        
                        # 修改Policy里的平台信息
                        result["Policy"]["SourceSystemID"] = "35020000032001000228"
                        result["Policy"]["SourceSystemName"] = "厦大非机动车违停检测系统平台"
                        print("finish the chaobiao!")
                    else:
                        continue
                    #redis_client.rpush('ebike_result', json.dumps(str(result)))
                    
                    

                    redis_client.rpush('ebike_result_xmu', json.dumps(str(result).replace("'", "\"")))
                    # result_list.append(result)

                    #  TODO: 保存信息用来自查，路径需要修改
                    #if len(result_list) > 10000:
                    #    np.save(f"/home/xxxy/Documents/EBike/check/result_list_{num_imgs[i] // 1000}.npy", result_list)
                    #    result_list = []
       except:
            pass



if __name__ == "__main__":
    load_batch_imgs()
