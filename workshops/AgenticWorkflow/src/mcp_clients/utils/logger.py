import logging
import os
from pathlib import Path

LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
LOG_NAME = 'client_logger'

log           = logging.getLogger(LOG_NAME)
log_formatter = logging.Formatter(LOG_FORMAT)

# Create logs directory in the project root
project_root = Path(__file__).parent.parent.parent.parent  # Go up to AgenticWorkflow directory
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

LOG_FILE_INFO = logs_dir / "log_client.log"
file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='w')
file_handler_info.setFormatter(log_formatter)
file_handler_info.setLevel(logging.INFO)
log.addHandler(file_handler_info)
log.info("Logger initialized.")