import logging
import os
from datetime import datetime

def setup_logging(log_dir: str, logger_name: str = 'Agent', log_level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup stream handler (console output)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Setup file handler (log file output)
    log_file_path = os.path.join(log_dir, 'agent.log')
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def get_log_dir() -> str:
    """
    Generate and return a consistent log directory path.
    """
    log_dir = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    screenshot_dir = os.path.join(log_dir, "screenshots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)
    return log_dir, screenshot_dir