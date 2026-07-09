import logging
import sys
from pathlib import Path

def setup_logger(name: str = "turbine_pipeline", log_file: str = "pipeline.log") -> logging.Logger:
    """
    configures and returns a centralized & thread-safe Logger.
    outputs log statements to both the console (sys.stdout) and a disk file.
    """
    logger = logging.getLogger(name)
    
    # if handlers are already configured, return the logger to prevent duplicate logs
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # formatter: ISO 8601 Timestamp | Severity Level | Module Line | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # handler 1: console (sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # handler 2: log file
    log_path = Path(log_file)
    try:
        # ensure directory path exists before creating the file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to initialize file logging handler. Error: {e}\n")
        
    return logger