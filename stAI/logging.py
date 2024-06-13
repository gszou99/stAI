import logging

# 重置日志配置，确保日志级别为INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取logger对象
logger = logging.getLogger(__name__)