import logging
from typing import Optional

_loggers = {}

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get logger for module.
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]
        
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)
        
    _loggers[name] = logger
    return logger
