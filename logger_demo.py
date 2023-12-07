## pip install loguru
import pytest
from loguru import logger

# logger.add("/logs/logger_demo.log",rotation="500MB",encoding="utf-8",enqueue=True,retention="10 days")
# logger.info('This is info information')

@pytest.fixture
def reportlog(pytestconfig):
    logging_plugin = pytestconfig.pluginmanager.getplugin("logging-plugin")
    handler_id = logger.add(logging_plugin.report_handler, format="{message}")
    yield
    logger.remove(handler_id)
