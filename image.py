import os
import shutil

def copy_and_rename_images(source_folder, destination_folder):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename == 'FaceSnap_01.jpg':
                source_path = os.path.join(root, filename)
                
                # 生成新的文件名
                new_filename = f"new_image_{len(os.listdir(destination_folder)) + 1}.jpg"
                
                # 拼接目标文件的完整路径
                destination_path = os.path.join(destination_folder, new_filename)
                
                # 复制文件并重新命名
                shutil.copy(source_path, destination_path)

# 调用函数并传入源文件夹和目标文件夹的路径
source_folder = "F:\\picture\\FaceSnap\\chan1"
destination_folder = "C:\\Users\ThinkPad\\Desktop\\json_to_txt\\image"
copy_and_rename_images(source_folder, destination_folder)
