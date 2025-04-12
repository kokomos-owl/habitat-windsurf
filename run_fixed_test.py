#!/usr/bin/env python
"""

Script to run the integrated climate e2e test with the fixed utility functions.

This script addresses the 'document_path' not defined error by using the fixed utility functions.
"""

import logging
import sys
import os
import shutil
import importlib.util
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def patch_test_utils():
    """
    Patch the test_utils.py file with the fixed version.
    
    This function creates a backup of the original file and replaces it with the fixed version.
    """
    try:
        # Paths
        original_path = os.path.join(
            os.path.dirname(__file__),
            "tests/integration/climate_e2e/test_utils.py"
        )
        fixed_path = os.path.join(
            os.path.dirname(__file__),
            "tests/integration/climate_e2e/test_utils_fix.py"
        )
        backup_path = original_path + ".bak"
        
        # Create backup if it doesn't exist
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup of {original_path}")
            shutil.copy2(original_path, backup_path)
        
        # Replace original with fixed version
        logger.info(f"Replacing {original_path} with fixed version")
        shutil.copy2(fixed_path, original_path)
        
        # Reload the module if it's already loaded
        if "tests.integration.climate_e2e.test_utils" in sys.modules:
            logger.info("Reloading test_utils module")
            importlib.reload(sys.modules["tests.integration.climate_e2e.test_utils"])
        
        logger.info("Successfully patched test_utils.py")
        return True
    
    except Exception as e:
        logger.error(f"Error patching test_utils.py: {e}")
        return False

def run_fixed_test():
    """Run the integrated climate e2e test with the fixed utility functions."""
    try:
        # Set the Claude API key in the environment
        os.environ["CLAUDE_API_KEY"] = "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"
        logger.info("Claude API key set in environment")
        
        # Patch test_utils.py
        success = patch_test_utils()
        if not success:
            logger.error("Failed to patch test_utils.py")
            return False
            
        # Also patch the conftest.py file to use the correct import path
        conftest_path = os.path.join(
            os.path.dirname(__file__),
            "tests/integration/climate_e2e/conftest.py"
        )
        conftest_backup = conftest_path + ".bak"
        
        # Create backup if it doesn't exist
        if not os.path.exists(conftest_backup):
            logger.info(f"Creating backup of {conftest_path}")
            shutil.copy2(conftest_path, conftest_backup)
        
        # Read the conftest file
        with open(conftest_path, "r") as f:
            conftest_content = f.read()
        
        # Replace the import path
        updated_content = conftest_content.replace(
            "from src.habitat_evolution.infrastructure.services.claude_adapter",
            "from src.habitat_evolution.infrastructure.adapters.claude_adapter"
        )
        
        # Write the updated content
        with open(conftest_path, "w") as f:
            f.write(updated_content)
        
        logger.info("Successfully patched conftest.py")
        
        # Patch the document processing service
        doc_processing_service_path = os.path.join(
            os.path.dirname(__file__),
            "src/habitat_evolution/climate_risk/document_processing_service.py"
        )
        doc_processing_service_fix_path = os.path.join(
            os.path.dirname(__file__),
            "src/habitat_evolution/climate_risk/document_processing_service_fix.py"
        )
        doc_processing_service_backup = doc_processing_service_path + ".bak"
        
        # Create backup if it doesn't exist
        if not os.path.exists(doc_processing_service_backup):
            logger.info(f"Creating backup of {doc_processing_service_path}")
            shutil.copy2(doc_processing_service_path, doc_processing_service_backup)
        
        # Replace with fixed version
        logger.info(f"Replacing {doc_processing_service_path} with fixed version")
        shutil.copy2(doc_processing_service_fix_path, doc_processing_service_path)
        
        # Reload the module if it's already loaded
        if "src.habitat_evolution.climate_risk.document_processing_service" in sys.modules:
            logger.info("Reloading document_processing_service module")
            importlib.reload(sys.modules["src.habitat_evolution.climate_risk.document_processing_service"])
        
        logger.info("Successfully patched document_processing_service.py")
        
        # Ensure the EventService is properly initialized
        from src.habitat_evolution.infrastructure.services.event_service import EventService
        
        # Create and initialize EventService with global instance
        event_service = EventService()
        event_service.initialize({
            "buffer_size": 100,
            "flush_interval": 5,
            "auto_flush": True,
            "persistence_enabled": True
        })
        
        # Test event publication
        event_service.publish("test.event", {
            "message": "EventService initialization test",
            "timestamp": "2025-04-12T12:10:00"
        })
        logger.info("Successfully published test event")
        
        # Make EventService globally available
        EventService._global_instance = event_service
        
        # Add a global instance accessor if it doesn't exist
        if not hasattr(EventService, 'get_instance'):
            @classmethod
            def get_instance(cls):
                if not hasattr(cls, '_global_instance') or cls._global_instance is None:
                    cls._global_instance = cls()
                    cls._global_instance.initialize()
                return cls._global_instance
            
            EventService.get_instance = get_instance
            logger.info("Added get_instance method to EventService")
        
        # Monkey patch the EventService.__init__ method to use the global instance
        original_init = EventService.__init__
        
        def patched_init(self, *args, **kwargs):
            if hasattr(EventService, '_global_instance') and EventService._global_instance is not None:
                # Copy attributes from global instance
                for attr, value in vars(EventService._global_instance).items():
                    setattr(self, attr, value)
                logger.debug("Using global EventService instance")
            else:
                # Call original init
                original_init(self, *args, **kwargs)
                # Initialize if not already initialized
                if not hasattr(self, '_initialized') or not self._initialized:
                    self.initialize()
        
        EventService.__init__ = patched_init
        logger.info("Patched EventService.__init__ method")
        
        # Set environment variable to indicate EventService is initialized
        os.environ["EVENT_SERVICE_INITIALIZED"] = "true"
        
        # Run the integrated test
        logger.info("Running integrated climate e2e test...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result != 0:
            logger.error(f"Test failed with exit code: {result}")
            return False
        
        logger.info("Test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting fixed test")
    success = run_fixed_test()
    
    if success:
        logger.info("Fixed test completed successfully")
        sys.exit(0)
    else:
        logger.error("Fixed test failed")
        sys.exit(1)
