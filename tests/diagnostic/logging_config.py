"""
Logging configuration for diagnostic tests.

This module provides a comprehensive logging configuration for diagnostic tests
that captures detailed information about component initialization and verification.
"""

import os
import logging
import logging.handlers
from datetime import datetime

def configure_diagnostic_logging(log_dir="logs"):
    """
    Configure logging for diagnostic tests.
    
    Args:
        log_dir: Directory to store log files
    
    Returns:
        Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate timestamp for log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"diagnostic_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Create file handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create a logger for this module
    logger = logging.getLogger("diagnostic")
    logger.info(f"Diagnostic logging configured. Log file: {log_file}")
    
    return logger

def log_component_status(logger, component, component_name):
    """
    Log detailed information about a component's status.
    
    Args:
        logger: Logger to use
        component: Component to log information about
        component_name: Name of the component
    """
    logger.info(f"Checking component: {component_name}")
    
    if component is None:
        logger.error(f"Component {component_name} is None")
        return
    
    # Log component type
    component_type = type(component).__name__
    logger.info(f"Component type: {component_type}")
    
    # Log initialization status
    if hasattr(component, '_initialized'):
        logger.info(f"Initialization status (_initialized): {component._initialized}")
    else:
        logger.info("No _initialized attribute found")
    
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        try:
            is_init = component.is_initialized()
            logger.info(f"Initialization status (is_initialized()): {is_init}")
        except Exception as e:
            logger.error(f"Error calling is_initialized(): {e}")
    else:
        logger.info("No is_initialized() method found")
    
    # Log attributes
    logger.debug(f"Attributes of {component_name}:")
    for attr in dir(component):
        if not attr.startswith('__') and not callable(getattr(component, attr)):
            try:
                value = getattr(component, attr)
                # Don't log large objects or sensitive data
                if isinstance(value, (str, int, float, bool, type(None))):
                    logger.debug(f"  {attr}: {value}")
                else:
                    logger.debug(f"  {attr}: {type(value)}")
            except Exception as e:
                logger.debug(f"  {attr}: Error accessing - {e}")
    
    # Log methods
    logger.debug(f"Methods of {component_name}:")
    for attr in dir(component):
        if not attr.startswith('__') and callable(getattr(component, attr)):
            logger.debug(f"  {attr}")

def log_exception_with_traceback(logger, e, message="Exception occurred"):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger to use
        e: Exception to log
        message: Message to log with the exception
    """
    import traceback
    logger.error(f"{message}: {e}")
    logger.error(traceback.format_exc())
