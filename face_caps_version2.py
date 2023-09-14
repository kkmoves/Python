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
rtsp_url = "rtsp://admin:nqiv123123@192.168.40.12/h264/ch1/main/av_stream"

# 打开视频流
video_capture = cv2.VideoCapture(rtsp_url)

# 保存参数
save_folder = os.path.join("output", current_month, current_day)
os.makedirs(save_folder, exist_ok=True)
save_interval = 115  # 每隔30帧保存一次
frame_count = 0

while True:
    try:
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
            if frame_count % save_interval == 0:
                save_path = os.path.join(save_folder, f"face_{i}_{frame_count}.jpg")
                cv2.imwrite(save_path, face_image)

        # 显示视频帧（可选）
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")

# 释放视频流
video_capture.release()
cv2.destroyAllWindows()
