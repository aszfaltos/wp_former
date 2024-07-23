import logging
from pathlib import Path


class WpFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    minimal_format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    MINIMAL_FORMATS = {
        logging.DEBUG: grey + minimal_format + reset,
        logging.INFO: grey + minimal_format + reset,
        logging.WARNING: yellow + minimal_format + reset,
        logging.ERROR: red + minimal_format + reset,
        logging.CRITICAL: bold_red + minimal_format + reset
    }

    def format(self, record):
        if hasattr(record, 'minimal'):
            log_fmt = self.MINIMAL_FORMATS.get(record.levelno)
        else:
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
            delete_prev = hasattr(record, 'delete_prev')
            if self.on_same_line and not same_line:
                stream.write(self.terminator)
            if self.on_same_line and delete_prev:
                stream.write('\r\033[0K')
            stream.write(msg)
            if same_line:
                self.on_same_line = True
            else:
                stream.write(self.terminator)
                self.on_same_line = False
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class Logger(logging.Logger):
    def __init__(self, name: str, level: int = logging.INFO):
        super().__init__(name, level)

        self.setLevel(logging.DEBUG)

        pch = ProgressConsoleHandler()
        pch.setLevel(level)

        logs_dir = Path('.').joinpath('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(f'logs/{name}.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        formatter = WpFormatter()
        pch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.propagate = False

        self.addHandler(pch)
        self.addHandler(fh)


