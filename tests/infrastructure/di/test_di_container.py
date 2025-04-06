"""
Tests for the DI container implementation.

These tests verify that the DI container works correctly, supporting the
principles of pattern evolution and co-evolution in Habitat.
"""

import sys
import os
import unittest
from typing import Protocol, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.habitat_evolution.infrastructure.di import DIContainer


# Define test interfaces and implementations
class ILogger(Protocol):
    """Test interface for a logger."""
    
    def log(self, message: str) -> None:
        """Log a message."""
        ...


class IRepository(Protocol):
    """Test interface for a repository."""
    
    def get_all(self) -> List[str]:
        """Get all items."""
        ...


class SimpleLogger:
    """Simple logger implementation."""
    
    def __init__(self):
        self.messages = []
        
    def log(self, message: str) -> None:
        """Log a message."""
        self.messages.append(message)


class ConfigurableLogger:
    """Logger with configuration options."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.messages = []
        
    def log(self, message: str) -> None:
        """Log a message with prefix."""
        self.messages.append(f"{self.prefix}{message}")


class Repository:
    """Repository implementation with logger dependency."""
    
    def __init__(self, logger: ILogger):
        self.logger = logger
        self.items = ["item1", "item2", "item3"]
        
    def get_all(self) -> List[str]:
        """Get all items and log the operation."""
        self.logger.log("Getting all items")
        return self.items


class DIContainerTests(unittest.TestCase):
    """Tests for the DI container."""
    
    def setUp(self):
        """Set up a new container for each test."""
        self.container = DIContainer()
        
    def test_register_and_resolve_singleton(self):
        """Test registering and resolving a singleton service."""
        # Register a singleton service
        self.container.register(ILogger, SimpleLogger, singleton=True)
        
        # Resolve the service twice
        logger1 = self.container.resolve(ILogger)
        logger2 = self.container.resolve(ILogger)
        
        # Verify that both references are the same instance
        self.assertIs(logger1, logger2)
        
        # Verify that the service works
        logger1.log("Test message")
        self.assertEqual(logger1.messages, ["Test message"])
        self.assertEqual(logger2.messages, ["Test message"])
        
    def test_register_and_resolve_transient(self):
        """Test registering and resolving a transient service."""
        # Register a transient service
        self.container.register(ILogger, SimpleLogger, singleton=False)
        
        # Resolve the service twice
        logger1 = self.container.resolve(ILogger)
        logger2 = self.container.resolve(ILogger)
        
        # Verify that both references are different instances
        self.assertIsNot(logger1, logger2)
        
        # Verify that the services work independently
        logger1.log("Message 1")
        logger2.log("Message 2")
        self.assertEqual(logger1.messages, ["Message 1"])
        self.assertEqual(logger2.messages, ["Message 2"])
        
    def test_register_with_factory(self):
        """Test registering a service with a factory function."""
        # Register a service with a factory
        def create_logger(container):
            return ConfigurableLogger(prefix="[TEST] ")
            
        self.container.register(ILogger, factory=create_logger)
        
        # Resolve the service
        logger = self.container.resolve(ILogger)
        
        # Verify that the factory was used
        self.assertIsInstance(logger, ConfigurableLogger)
        self.assertEqual(logger.prefix, "[TEST] ")
        
        # Verify that the service works
        logger.log("Factory message")
        self.assertEqual(logger.messages, ["[TEST] Factory message"])
        
    def test_dependency_injection(self):
        """Test automatic dependency injection."""
        # Register both services
        self.container.register(ILogger, SimpleLogger)
        self.container.register(IRepository, Repository)
        
        # Resolve the repository
        repo = self.container.resolve(IRepository)
        
        # Verify that the logger was injected
        self.assertIsInstance(repo.logger, SimpleLogger)
        
        # Verify that both services work together
        items = repo.get_all()
        self.assertEqual(items, ["item1", "item2", "item3"])
        self.assertEqual(repo.logger.messages, ["Getting all items"])
        
    def test_reset_container(self):
        """Test resetting the container."""
        # Register and resolve a singleton service
        self.container.register(ILogger, SimpleLogger)
        logger1 = self.container.resolve(ILogger)
        
        # Reset the container
        self.container.reset()
        
        # Resolve the service again
        logger2 = self.container.resolve(ILogger)
        
        # Verify that the instances are different
        self.assertIsNot(logger1, logger2)
        
    def test_clear_container(self):
        """Test clearing the container."""
        # Register a service
        self.container.register(ILogger, SimpleLogger)
        
        # Clear the container
        self.container.clear()
        
        # Verify that the service is no longer registered
        with self.assertRaises(KeyError):
            self.container.resolve(ILogger)


if __name__ == "__main__":
    unittest.main()
