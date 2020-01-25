from logging import getLogger, Formatter, DEBUG, FileHandler, StreamHandler


def get_my_logger(name):
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    fh = FileHandler('log/default.log', mode='a')
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
