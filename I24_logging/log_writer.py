import logging
import socket
from logging.handlers import SysLogHandler
from logstash_async.handler import AsynchronousLogstashHandler
from logstash_async.formatter import LogstashFormatter
import ecs_logging
import sys
import os

from typing import Union, Mapping

levels = {'CRITICAL': logging.CRITICAL, 'ERROR': logging.ERROR, 'WARNING': logging.WARNING,
          'INFO': logging.INFO, 'DEBUG': logging.DEBUG, None: None}


class MaxLevelFilter(object):
    """
    Filter for keeping log records of a given level or LOWER (as opposed to the normal 'or higher' functionality).
    Does not inherit from logging.Filter, since we need our own __init__ to keep track of the max level.
    Inspired from: https://pythonexamples.org/python-logging-info/#3
    """
    def __init__(self, level):
        """
        Establish the filter with maximum log level to keep.
        :param level: maximum logging level (e.g., logging.INFO) to allow through the filter
        """
        self.__max_level = level

    def __call__(self, log_record: logging.LogRecord) -> bool:
        """
        Filter function, implemented as the direct call of this object.
        :param log_record: logging.LogRecord that contains all the relevant fields and functionality.
        :return:
        """
        return log_record.levelno <= self.__max_level


class ExtraLogger(logging.Logger):
    """
    Subclass of logging.Logger that adds "extra" log record information passed as a dictionary as 1) unpacked individual
        LogRecord attributes (default behavior) and 2) as a single attribute that contains the entire dictionary. This
        feature is needed in order to unify the logging interface between different code modules that will want to
        include different "extra" fields depending on context.
    Inspired from: https://devdreamz.com/question/710484-python-logging-logger-overriding-makerecord
    """
    def makeRecord(self, name: str, level: int, fn: str, lno: int, msg: object, args, exc_info,
                   func: Union[str, None] = None,  extra: Union[Mapping[str, object], None] = None,
                   sinfo: Union[str, None] = None) -> logging.LogRecord:
        """
        Overrides `makeRecord` in logging.Logger in order to add a single feature: add the attribute 'extra' to each
            LogRecord that is created and set its value as the entire "extra" dictionary that is passed to the log
            function that initiated the record creation. The "extra" dictionary still gets unpacked and added as
            individual attributes through the call to super.makeRecord(...).
        :param name: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param level: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param fn: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param lno: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param msg: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param args: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param exc_info: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param func: passed straight to the LogRecord factory, which by default is the LogRecord class
        :param extra: a dictionary of extra log information that is contextual to the code module logging call
        :param sinfo: passed straight to the LogRecord factory, which by default is the LogRecord class
        :return: LogRecord with the desired 'extra' attribute and unpacked "extra" values
        """
        # Make the call to the normal `makeRecord` function, which will do the default behavior
        rv = super(ExtraLogger, self).makeRecord(name=name, level=level, fn=fn, lno=lno, msg=msg, args=args,
                                                 exc_info=exc_info, func=func, extra=extra, sinfo=sinfo)
        # Also add the complete "extra" dictionary as an attribute
        rv.__dict__['extra'] = extra
        return rv


class I24Logger:
    """
    This unified interface is used to abstract log setup from other code modules,
        which we want to have consistent behavior.
    """

    #def __init__(self, server_id: Union[str, int], environment: str, owner_process_name: str,
    #             owner_process_id: int, owner_parent_name: str = None,
    #             connect_logstash: bool = False, connect_file: bool = False,
    #             connect_syslog: bool = False, connect_console: bool = False,
    #             logstash_host: str = None, logstash_port: int = None, file_path: str = None,
    #             syslog_location: Union[tuple[str, int], str] = None, all_log_level: str = 'DEBUG',
    #             logstash_log_level: Union[str, None] = None, file_log_level: Union[str, None] = None,
    #             syslog_log_level: Union[str, None] = None, console_log_level: Union[str, None] = None):

    # Python 3.8.10 compatibility mode
    def __init__(self, server_id: str = None, environment: str = None, owner_process_name: str = None,
                 owner_process_id: int = -1, owner_parent_name: str = None,
                 connect_logstash: bool = False, connect_file: bool = False,
                 connect_syslog: bool = False, connect_console: bool = False,
                 logstash_host: str = None, logstash_port: int = None, file_path: str = None,
                 syslog_location = None, all_log_level: str = 'DEBUG',
                 logstash_log_level = None, file_log_level = None,
                 syslog_log_level = None, console_log_level = None):


        """
        Constructor of the persistent logging interface. It establishes a custom multi-destination logger with the
            option to log different levels to different destinations.
        :param server_id: Identifier (string or int) of the server on which this logger is running.
        :param environment: Software environment for this logger (e.g., production, development). Not currently used
            to set logging handlers in this object, but the two can be controlled jointly where this logger is created.
        :param owner_process_name: The name (official or unofficial) of the process that created/owns this logger.
        :param owner_process_id: Process ID (PID) of the process that created/owns this logger.
        :param owner_parent_name: Parent process of the owner of this logger; useful for tracking hierarchy in logs.
        :param connect_logstash: True/False to connect to Logstash via asynchronous handler.
        :param connect_file: True/False to connect a simple log file (non-rotating) to this logger. If multiple loggers
            are instantiated, multiple files will be produced and need to be differentiated by `file_path`.
        :param connect_syslog: True/False to connect to the host computer's syslog via TCP Socket Stream.
        :param connect_console: True/False to connect to the STDOUT and STDERR available via `sys` package.
        :param logstash_host: Hostname of the Logstash server.
        :param logstash_port: Port number of the Logstash server.
        :param file_path: Path (absolute or relative) and file name of the log file to write; directories not created.
        :param all_log_level: Available to set a global log level across all handlers; overridden by handler-specific.
        :param logstash_log_level: Logstash log level; overrides `all_log_level`.
        :param file_log_level: File log level; overrides `all_log_level`.
        :param syslog_log_level: Syslog log level; overrides `all_log_level`.
        :param console_log_level: Console log level; overrides `all_log_level`.
        """
                
        self._owner_pid = owner_process_id if owner_process_id >= 0 else os.getpid()
        
        self._owner_name = owner_process_name if owner_process_name is not None else 'PID-{}'.format(self._owner_pid)

        self._server_id = server_id if server_id is not None else socket.gethostname()
        
        self._environment = environment if environment is not None else 'DEF_ENV'                    

        self._owner_parent_name = owner_parent_name
        
        # The name of the logger we create with this class will have this name, possibly with parent.child syntax.
        self._name = (owner_parent_name + '.' if owner_parent_name is not None else '') + self._owner_name
        
        self._logfile_path = file_path if file_path is not None else '{}_{}.log'.format(self._owner_name, self._owner_pid)
                
        
        self._default_logger_extra = {'serverid': self._server_id, 'environment': self._environment,
                                      'ownername': self._owner_name, 'ownerpid': self._owner_pid,
                                      'parentname': self._owner_parent_name}

        if not all([ll in levels.keys() for ll in
                    (logstash_log_level, file_log_level, syslog_log_level, console_log_level)]):
            raise ValueError("Invalid log level specified. Use: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', None.")
        self._log_levels = {'logstash': (levels[logstash_log_level] if logstash_log_level is not None
                                         else levels[all_log_level]),
                            'file': (levels[file_log_level] if file_log_level is not None
                                     else levels[all_log_level]),
                            'syslog': (levels[syslog_log_level] if syslog_log_level is not None
                                       else levels[all_log_level]),
                            'console': (levels[console_log_level] if console_log_level is not None
                                        else levels[all_log_level]),
                            }
        if connect_logstash is True and self._log_levels['logstash'] is None:
            raise ValueError("Logstash logging activated, but no log level specified during construction.")
        if connect_file is True and self._log_levels['file'] is None:
            raise ValueError("File logging activated, but no log level specified during construction.")
        if connect_syslog is True and self._log_levels['syslog'] is None:
            raise ValueError("Syslog logging activated, but no log level specified during construction.")
        if connect_console is True and self._log_levels['console'] is None:
            raise ValueError("Console logging activated, but no log level specified during construction.")

        if connect_logstash is True and (logstash_host is None or logstash_port is None):
            raise ValueError("Logstash logging activated, but no connection information given (host and port).")
        if connect_file is True and (self._logfile_path is None or self._logfile_path == ''):
            raise ValueError("File logging activated, but no file path given.")
        if connect_syslog is True and syslog_location is None:
            raise ValueError("Syslog logging activated, but no location (path or host/port tuple) given.")

        self._logstash_host, self._logstash_port = logstash_host, logstash_port        
        self._syslog_location = syslog_location

        logging.setLoggerClass(ExtraLogger)
        self._logger = logging.getLogger(self._name)
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)

        if connect_logstash is True:
            self._setup_logstash()
        if connect_file is True:
            self._setup_file()
        if connect_syslog is True:
            self._setup_syslog()
        if connect_console is True:
            self._setup_stdout()

    def _setup_logstash(self):
        """
        Attaches a Logstash asynchronous handler, which executes transactions without blocking primary code. Uses
            connection information given in the I24Logger constructor. Log level is also set in the constructor.
            Formatter is currently the LogstashFormatter with only `message_type='python-logstash'`, which appears
             to be purely cosmetic and not a behavior change.
        :return: None
        """
        # Set database_path to None to use in-memory caching.
        lsth = AsynchronousLogstashHandler(self._logstash_host, self._logstash_port, database_path=None)
        lsth.setLevel(self._log_levels['logstash'])
        # Not using the "extra" feature of the LogstashFormatter, since we already have the desired merge behavior
        #   in our own logger object.
        lstf = LogstashFormatter(message_type='python-logstash', extra_prefix=None)
        lsth.setFormatter(lstf)
        self._logger.addHandler(lsth)

    def _setup_syslog(self, elastic_format: bool = False):
        """
        Attaches a syslog handler for this machine. The path of the syslog is needed in the I24Logger constructor, since
            platforms have different destinations (e.g., Mac appears to be '/var/run/syslog' and Linux is usually
            '/var/log/syslog'). There are two formatting options: ECS, which makes logs easily importable into Elastic,
            and a default time/level/name/message/extra line format.
        :param elastic_format: True/False to use Elastic-compatible formatting.
        :return: None
        """
        sysh = SysLogHandler(address=self._syslog_location, socktype=socket.SOCK_STREAM)
        sysh.setLevel(self._log_levels['syslog'])
        if elastic_format is True:
            ecsfmt = ecs_logging.StdlibFormatter()
            sysh.setFormatter(ecsfmt)
        else:
            exfmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s | %(extra)s')
            sysh.setFormatter(exfmt)
        self._logger.addHandler(sysh)

    def _setup_file(self, elastic_format: bool = False):
        """
        Attaches a non-rotating file handler. The file path is given during I24Logger construction. Formatting is by
            default a simple line of information that is easily readble, but can also be made compatible with Elastic.
        :param elastic_format: True/False to use Elastic-compatible formatting.
        :return: None
        """
        flh = logging.FileHandler(filename=self._logfile_path)
        flh.setLevel(self._log_levels['file'])
        if elastic_format is True:
            ecsfmt = ecs_logging.StdlibFormatter()
            flh.setFormatter(ecsfmt)
        else:
            exfmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s | %(extra)s')
            flh.setFormatter(exfmt)
        self._logger.addHandler(flh)

    def _setup_stdout(self, stdout_max_level=logging.INFO):
        """
        Attaches a STDOUT/STDERR handler. Messages at INFO/DEBUG level are handled through STDOUT and WARNING and higher
            are handled through STDERR in order to take advantage of typically built-in formatting (e.g., red text).
            That filtering is accomplished through the custom MaxLevelFilter, which can be set with `stdout_max_level`.
        :param stdout_max_level: Option to set STDOUT max log level, everything higher goes to STDERR. *Not currently
            configurable/implemented in constructor.*
        :return: None
        """
        if stdout_max_level not in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            raise ValueError("Must provide valid logging level for maximum log level to STDOUT.")
        csfmt = logging.Formatter('%(levelname)s | %(name)s | %(message)s | %(extra)s')
        if self._log_levels['console'] <= logging.INFO:
            outh = logging.StreamHandler(stream=sys.stdout)
            outh.setLevel(self._log_levels['console'])
            outh.addFilter(filter=MaxLevelFilter(level=stdout_max_level))
            outh.setFormatter(csfmt)
            self._logger.addHandler(outh)
        errh = logging.StreamHandler(stream=sys.stderr)
        errh.setLevel(max(self._log_levels['console'], logging.WARNING))
        errh.setFormatter(csfmt)
        self._logger.addHandler(errh)

    def debug(self, message: Union[str, BaseException], extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the DEBUG level, which is the lowest order of precedence.
        Anything given in `extra` is merged with the values in the I24Logger constructor. This is the location in
            which contextual information should be passed. This allows, particularly in LogStash, this information
            to be separated automatically from the log message and to maintain its type. For example, one might include
            information about processing rate (e.g., frames per second, trajectories per minute) or status of monitored
            assets (e.g., cameras).
        In order to log an exception traceback, pass the exception or a message as `message`, and set `exc_info`=True;
            or write a message and pass the exception object as `exc_info`. Support is available for just setting
            `exc_info`=True and letting `logging` automatically gather the traceback, but it is recommended to be
            explicit about including the exception.
        ```
        try:
            raise ValueError("Parameter invalid.")
        except ValueError as e:
            my_logger.warning(e, exc_info=True)                     # Option 1
            my_logger.warning("Got an exception!", exc_info=e)      # Option 2
        ```
        :param message: Either a log message as a string, or an exception.
        :param extra: Dictionary of extra contextual information about the log message.
        :param exc_info: True/False to automatically include exception info, or the exception itself (recommended).
        :return: None
        """
        extra = extra if extra is not None else {}
        self._logger.debug(message, extra={**self._default_logger_extra, **extra}, exc_info=exc_info)

    def info(self, message: Union[str, BaseException], extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the INFO level. See .debug(...) for more information.
        """
        extra = extra if extra is not None else {}
        self._logger.info(message, extra={**self._default_logger_extra, **extra}, exc_info=exc_info)

    def warning(self, message: Union[str, BaseException], extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the WARNING level. See .debug(...) for more information.
        """
        extra = extra if extra is not None else {}
        self._logger.warning(message, extra={**self._default_logger_extra, **extra}, exc_info=exc_info)

    def error(self, message: Union[str, BaseException], extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the ERROR level. See .debug(...) for more information.
        """
        extra = extra if extra is not None else {}
        self._logger.error(message, extra={**self._default_logger_extra, **extra}, exc_info=exc_info)

    def critical(self, message: Union[str, BaseException], extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the CRITICAL level. See .debug(...) for more information.
        """
        extra = extra if extra is not None else {}
        self._logger.critical(message, extra={**self._default_logger_extra, **extra}, exc_info=exc_info)

    def log(self, level: str, message: Union[str, BaseException],
            extra: Union[dict, None] = None, exc_info: bool = False):
        """
        Logs a message at the level specified in `level` (as a string). Otherwise, behavior is the same as .debug(...).
        """
        level_upper = level.upper()
        if level_upper == 'DEBUG':
            self.debug(message=message, extra=extra, exc_info=exc_info)
        elif level_upper == 'INFO':
            self.info(message=message, extra=extra, exc_info=exc_info)
        elif level_upper == 'WARNING':
            self.warning(message=message, extra=extra, exc_info=exc_info)
        elif level_upper == 'ERROR':
            self.error(message=message, extra=extra, exc_info=exc_info)
        elif level_upper == 'CRITICAL':
            self.critical(message=message, extra=extra, exc_info=exc_info)
