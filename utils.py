import logging

def get_logger(logfile_name):
    simple_formatter = logging.Formatter("[%(name)s] %(message)s")
    complex_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logfile_name)
    file_handler.setFormatter(complex_formatter)
    file_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    return root_logger