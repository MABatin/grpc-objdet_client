import os
from loguru import logger
import sys

logger.remove(0)
logger.add(sys.stdout, level="INFO",
           format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [<level>{level}</level>] | <level>{message}</level>",
           diagnose=False)