import atexit
import logging

def create_logger(log_filename, file_dir):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{file_dir}/{log_filename}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    def flush_logs():
        for handler in logger.handlers:
            handler.flush()
    atexit.register(flush_logs)
    return logger

