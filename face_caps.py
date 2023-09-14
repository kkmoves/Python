import cv2
import face_recognition
import os
import logging
import datetime

# 获取当月和当日的日期
current_date = datetime.datetime.now()
current_month = current_date.strftime("%Y-%m")
current_day = current_date.strftime("%Y-%m-%d")

# 创建日志文件夹
log_folder = os.path.join("logs", current_month, current_day)
os.makedirs(log_folder, exist_ok=True)

# 配置日志记录
log_file = os.path.join(log_folder, "log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# RTSP视频流地址
rtsp_url = "rtsp://admin:nqiv123123@192.168.40.3/h264/ch1/main/av_stream"

# 打开视频流
video_capture = cv2.VideoCapture(rtsp_url)

while True:
    # 读取视频帧
    ret, frame = video_capture.read()

    if not ret:
        break

    # 使用face_recognition库检测人脸
    face_locations = face_recognition.face_locations(frame)

    # 输出每一帧的人脸检测结果
    print(f"Found {len(face_locations)} face(s) in this frame.")
    logging.info(f"Found {len(face_locations)} face(s) in this frame.")

    # 在帧上绘制人脸框
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # 保存检测到的人脸
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        cv2.imwrite(f"face_{i}.jpg", face_image)

    # 显示视频帧（可选）
    # cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# 示例日志消息
# logging.info("This is a log message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.debug("Log 日志测试.")

# 释放视频流和关闭窗口
video_capture.release()
cv2.destroyAllWindows()
























# import cv2
# import face_recognition

# # RTSP视频流地址
# rtsp_url = "rtsp://admin:nqiv123123@192.168.40.3/h264/ch1/main/av_stream"

# # 打开视频流
# video_capture = cv2.VideoCapture(rtsp_url)

# while True:
#     # 读取视频帧
#     ret, frame = video_capture.read()

#     if not ret:
#         break

#     # 使用face_recognition库检测人脸
#     face_locations = face_recognition.face_locations(frame)

#     # 保存检测到的人脸
#     for i, face_location in enumerate(face_locations):
#         top, right, bottom, left = face_location
#         face_image = frame[top:bottom, left:right]
#         cv2.imwrite(f"face_{i}.jpg", face_image)

#     # 显示视频帧（可选）
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放视频流和关闭窗口
# video_capture.release()
# cv2.destroyAllWindows()
