import os
import shutil
import time

# 开始计时
start_time = time.time()

# 在这里放置你要计时的任务或代码块
# 例如，你可以运行一些耗时的操作
for _ in range(1000000):
    pass

# 结束计时
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time

# 打印执行时间
# print(f"任务执行花费时间: {execution_time:.4f} 秒")

# 源文件夹路径
source_folder = 'C:\\Users\\ThinkPad\Desktop\\New-model-Testing'

# 目标文件夹路径
dest_folder_buchaobiao = 'C:\\Users\ThinkPad\\Desktop\\New-model-Testing\\2'
dest_folder_chaobiao = 'C:\\Users\\ThinkPad\Desktop\\New-model-Testing\\1'

# 关键字
keyword_buchaobiao = 'buchaobiao'
keyword_chaobiao = 'chaobiao'

# 遍历源文件夹中的文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        file_path = os.path.join(root, file)
        
        # 检查文件名是否包含关键字'buchaobiao'或'chaobiao'
        if keyword_buchaobiao in file:
            # 移动到指定文件夹D:/2
            shutil.move(file_path, os.path.join(dest_folder_buchaobiao, file))
        elif keyword_chaobiao in file:
            # 移动到指定文件夹D:/1
            shutil.move(file_path, os.path.join(dest_folder_chaobiao, file))
