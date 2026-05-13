import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from TMMKG.utils.config import project_path

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)


def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str = "app.log",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """初始化项目日志配置。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler._tmmkg_handler = True

    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler._tmmkg_handler = True

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def setup_logging_from_config(config: dict, program_name: str) -> None:
    """根据 YAML 配置为指定程序初始化日志。"""
    logging_config = config.get("logging", {})
    program_config = logging_config.get("programs", {}).get(program_name, {})

    log_dir = program_config.get("dir", logging_config.get("dir", "logs"))
    setup_logging(
        log_dir=project_path(log_dir),
        log_file=program_config.get(
            "file",
            logging_config.get("file", "app.log"),
        ),
        level=program_config.get("level", logging_config.get("level", "INFO")),
        max_bytes=program_config.get(
            "max_bytes",
            logging_config.get("max_bytes", 10 * 1024 * 1024),
        ),
        backup_count=program_config.get(
            "backup_count",
            logging_config.get("backup_count", 5),
        ),
        log_format=program_config.get(
            "format",
            logging_config.get("format", DEFAULT_LOG_FORMAT),
        ),
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(name)
