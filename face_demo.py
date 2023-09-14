import face_recognition
image = face_recognition.load_image_file("D:\\Downloads\\Documents\\Python\\output\\2023-09\\2023-09-14\\1694669056033.jpg")
# face_locations 定位人脸坐标
face_locations = face_recognition.face_locations(image)
print(face_locations)

# 人脸关键点
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)