import sys
from datetime import datetime

LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class Logger(object):
    def __init__(self, minimum_level="WARNING", maximum_level="CRITICAL", log_path=r"./log.txt"):
        self.__check_setting(minimum_level, maximum_level, log_path)

        self.__minimum_level = LEVEL.index(minimum_level)
        self.__maximum_level = LEVEL.index(maximum_level)
        self.log_path = log_path
        self._initialized = False
        self.terminal = sys.stdout

    def current_setting(self):
        assert self.__minimum_level is not None and self.__maximum_level is not None, \
            "Please set minimum level and maximum level."
        print(f"Current Setting:\n"
              f"       Minimum Level: {LEVEL[self.__minimum_level]}\n"
              f"       Maximum Level: {LEVEL[self.__maximum_level]}\n"
              f"       Log File Path: {self.log_path}\n")

    def log(self, level, msg):
        if level not in LEVEL:
            raise ValueError("Unsupported log level. Supported levels: CRITICAL, ERROR, WARNING, INFO, DEBUG.")
        level_index = LEVEL.index(level)
        if self.__minimum_level <= level_index <= self.__maximum_level:
            current_time = datetime.now().strftime("%m-%d %H:%M:%S")
            log_entry = f"[{current_time}] [{level}]: {msg}\n"
            self.__record_log_txt_file(log_entry)
        else:
            return None

    def __record_log_txt_file(self, log_entry):
        try:
            with open(self.log_path, "a+") as log_file:
                log_file.write(log_entry)
        except FileNotFoundError:
            print("Log File Not Found")

    def __check_setting(self, minimum_level, maximum_level, log_path):
        assert minimum_level in LEVEL and maximum_level in LEVEL, \
            "Level are not supported. Supported levels: CRITICAL, ERROR, WARNING, INFO, DEBUG."
        assert type(log_path) is str, "Log path must be a string."

    def DEBUG_log(self, msg): self.log("DEBUG", msg)

    def INFO_log(self, msg): self.log("INFO", msg)

    def ERROR_log(self, msg): self.log("ERROR", msg)

    def CRITICAL_log(self, msg): self.log("CRITICAL", msg)

    def WARNING_log(self, msg): self.log("WARNING", msg)


def rawCharMatching(char):
    pass
