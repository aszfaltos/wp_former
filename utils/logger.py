import logging


class WpFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ProgressConsoleHandler(logging.StreamHandler):
    """
    A handler class which allows the cursor to stay on
    one line for selected messages.
    """
    on_same_line = False

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            same_line = hasattr(record, 'same_line')
            if self.on_same_line and not same_line:
                stream.write(self.terminator)
            stream.write(msg)
            if same_line:
                stream.write('... ')
                self.on_same_line = True
            else:
                stream.write(self.terminator)
                self.on_same_line = False
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class Logger:
    def __init__(self, app_name: str, log_level: int = logging.INFO):
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(log_level)

        pch = ProgressConsoleHandler()
        fh = logging.FileHandler(f'{app_name}.log')

        formatter = WpFormatter()
        pch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.logger.addHandler(pch)
        self.logger.addHandler(fh)

    def debug(self, message: str, same_line: bool = False, email: bool = False):
        self.logger.debug(message, extra={'same_line': same_line})

    def info(self, message: str, same_line: bool = False, email: bool = False):
        self.logger.info(message, extra={'same_line': same_line})

    def warning(self, message: str, same_line: bool = False, email: bool = False):
        self.logger.warning(message, extra={'same_line': same_line})

    def error(self, message: str, same_line: bool = False, email: bool = False):
        self.logger.error(message, extra={'same_line': same_line})

    def critical(self, message: str, same_line: bool = False, email: bool = False):
        self.logger.critical(message, extra={'same_line': same_line})

    def _send_email(self, message: str):
        pass
