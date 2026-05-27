import logging
import os
import sys

from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs")

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# CREATE LOGGER
logger = logging.getLogger("my_project")

logger.setLevel(logging.INFO)

# Prevent duplicate logs
logger.handlers.clear()

# FORMATTER
formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# FILE HANDLER
file_handler = logging.FileHandler(LOG_FILE_PATH)

file_handler.setFormatter(formatter)

# CONSOLE HANDLER
console_handler = logging.StreamHandler(sys.stdout)

console_handler.setFormatter(formatter)

# ADD HANDLERS
logger.addHandler(file_handler)

logger.addHandler(console_handler)