import subprocess
import time
import logging
import psutil

#######################################
#
# 违停算法异常停止,监测算法PID 重启算法
#
#######################################


# 定义要查询的进程名
process_name = "startdetect.py"
script_path = "/home/xxxy/Document/illegalpark/stsrtdetect.py"
conda_env = "yolo"

# 日志
log_file = "process_monitor.log"

# 配置日志
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

while True:
    try:
        result = subprocess.check_output(["ps", "-ef"], universal_newlines=True)

        # 获取所有进程信息
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'] == process_name:
                logging.info(f"进程 {process_name} 存在，PID: {process.info['pid']}")
                break

        # 在输出结果中查找进程名
        if process_name in result:
            logging.info(f"进程 {process_name} 存在")
        else:
            logging.warning(f"进程 {process_name} 不存在，将执行启动命令")
            # 在这里执行启动命令，可以使用之前提到的方式
            # subprocess.Popen(f"conda run -n yolo python {script_path} &", executable='/bin/bash', shell=True)
            subprocess.Popen(f"tmux send-keys -t yolo 'python {script_path} &' C-m", shell=True)
            # cmd = f"source activate {conda_env} && python {script_path} &"

        # 每隔一段时间重新查询
        time.sleep(60)  # 60秒（一分钟）查询一次，可以根据需要调整间隔时间
    except subprocess.CalledProcessError:
        logging.error("执行查询命令时出错")
