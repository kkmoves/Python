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

# 示例日志消息
logging.info("This is a log message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.debug("Log 日志测试.")
