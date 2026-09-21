import logging
import os
import sys
from datetime import datetime

def setup_logging(script_name, side=None):
    """
    Configures a dual-handler logger (File + Console) using relative paths
    from the project root.
    
    Args:
        script_name (str): The name of the calling script (e.g., 'analyze_rms').
        side (str, optional): 'left' or 'right' to distinguish camera logs.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # 1. Create logs directory in the current working directory (project root)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Define File Name: [SCRIPT]_[SIDE]_[TIMESTAMP].log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    side_str = f"_{side}" if side else ""
    log_filename = f"{script_name}{side_str}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # 3. Configure Logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Clear handlers to avoid duplicate logs in interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter: Time | Script | Level | Message
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger
