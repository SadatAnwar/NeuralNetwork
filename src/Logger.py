import climate


class Logger(object):
    def __init__(self, name):
        self.__name__ = name
        name = '.'.join([self.__class__.__name__])
        self.logger = climate.get_logger(name)

    def debug(self, str):
        self.logger.debug(str)

    def info(self, str):
        self.logger.info(msg=str)

    def warn(self, str):
        self.logger.warn(msg=str)

    def error(self, str):
        self.logger.error(msg=str)
