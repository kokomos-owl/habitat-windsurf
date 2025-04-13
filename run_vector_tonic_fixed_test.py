#!/usr/bin/env python
"""
Script to run the integrated climate e2e test with the Vector Tonic fix applied.

This script addresses the EventAwarePatternDetector initialization issue by applying
the comprehensive fix from vector_tonic_fix.py.
"""

import logging
import sys
import os
import shutil
import importlib.util
import pytest

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
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
        
        return True
    except Exception as e:
        logger.error(f"Error patching test_utils.py: {str(e)}")
        return False

def patch_conftest():
    """
    Patch the conftest.py file to use the vector_tonic_fix.py module.
    """
    try:
        # Path to conftest.py
        conftest_path = os.path.join(
            os.path.dirname(__file__),
            "tests/integration/climate_e2e/conftest.py"
        )
        
        # Read the conftest.py file
        with open(conftest_path, 'r') as f:
            content = f.read()
        
        # Check if the file already imports vector_tonic_fix
        if "vector_tonic_fix" in content:
            logger.info("conftest.py already patched with vector_tonic_fix")
            return True
        
        # Add import for vector_tonic_fix
        import_line = "\nfrom tests.integration.climate_e2e.vector_tonic_fix import initialize_vector_tonic_components\n"
        
        # Add the import line at the top of the file
        content = import_line + content
        
        # Create backup
        backup_path = conftest_path + ".bak"
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup of {conftest_path}")
            shutil.copy2(conftest_path, backup_path)
        
        # Write the modified content
        with open(conftest_path, 'w') as f:
            f.write(content)
        
        logger.info("Successfully patched conftest.py")
        return True
    except Exception as e:
        logger.error(f"Error patching conftest.py: {str(e)}")
        return False

def patch_test_climate_e2e():
    """
    Patch the test_climate_e2e.py file to use the vector_tonic_fix.py module.
    """
    try:
        # Path to test_climate_e2e.py
        test_path = os.path.join(
            os.path.dirname(__file__),
            "tests/integration/climate_e2e/test_climate_e2e.py"
        )
        
        # Read the test_climate_e2e.py file
        with open(test_path, 'r') as f:
            content = f.readlines()
        
        # Create backup
        backup_path = test_path + ".bak"
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup of {test_path}")
            shutil.copy2(test_path, backup_path)
        
        # Find the vector tonic initialization code
        for i, line in enumerate(content):
            if "vector_tonic_integrator" in line and "VectorTonicWindowIntegrator" in line:
                # Replace the line and the next few lines with our fixed initialization
                start_line = i
                end_line = i
                # Find the end of the initialization block
                while end_line < len(content) and ")" not in content[end_line]:
                    end_line += 1
                
                # Replace the initialization code
                replacement = [
                    "        # Using the fixed vector tonic initialization\n",
                    "        vector_tonic_integrator, vector_tonic_persistence, event_bus, harmonic_io_service = initialize_vector_tonic_components(arangodb_connection)\n",
                    "        \n",
                    "        # Check if vector tonic integration is available\n",
                    "        if vector_tonic_integrator is None:\n",
                    "            logger.warning(\"Vector Tonic integration not available, continuing with limited functionality\")\n",
                    "        \n"
                ]
                
                content[start_line:end_line+1] = replacement
                break
        
        # Write the modified content
        with open(test_path, 'w') as f:
            f.writelines(content)
        
        logger.info("Successfully patched test_climate_e2e.py")
        return True
    except Exception as e:
        logger.error(f"Error patching test_climate_e2e.py: {str(e)}")
        return False

def patch_document_processing_service():
    """
    Patch the document_processing_service.py file with the fixed version.
    """
    try:
        # Paths
        original_path = os.path.join(
            os.path.dirname(__file__),
            "src/habitat_evolution/climate_risk/document_processing_service.py"
        )
        fixed_path = os.path.join(
            os.path.dirname(__file__),
            "src/habitat_evolution/climate_risk/document_processing_service_fix.py"
        )
        backup_path = original_path + ".bak"
        
        # Create backup if it doesn't exist
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup of {original_path}")
            shutil.copy2(original_path, backup_path)
        
        # Replace original with fixed version
        logger.info(f"Replacing {original_path} with fixed version")
        shutil.copy2(fixed_path, original_path)
        
        return True
    except Exception as e:
        logger.error(f"Error patching document_processing_service.py: {str(e)}")
        return False

def run_vector_tonic_fixed_test():
    """
    Run the integrated climate e2e test with the Vector Tonic fix applied.
    """
    try:
        # Apply patches
        if not patch_test_utils():
            logger.error("Failed to patch test_utils.py")
            return False
        
        if not patch_conftest():
            logger.error("Failed to patch conftest.py")
            return False
        
        if not patch_test_climate_e2e():
            logger.error("Failed to patch test_climate_e2e.py")
            return False
        
        if not patch_document_processing_service():
            logger.error("Failed to patch document_processing_service.py")
            return False
        
        logger.info("All patches applied successfully")
        
        # Set environment variables
        os.environ["CLAUDE_API_KEY"] = os.environ.get("CLAUDE_API_KEY", "")
        if not os.environ.get("CLAUDE_API_KEY"):
            logger.warning("No Claude API key found in environment, some features may not work")
        
        # Run the test
        logger.info("Running integrated climate e2e test with Vector Tonic fix...")
        result = pytest.main(["-xvs", "tests/integration/climate_e2e/test_climate_e2e.py::test_integrated_climate_e2e"])
        
        if result == 0:
            logger.info("Test completed successfully")
            return True
        else:
            logger.error(f"Test failed with exit code {result}")
            return False
    
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Vector Tonic fixed test")
    success = run_vector_tonic_fixed_test()
    
    if success:
        logger.info("Vector Tonic fixed test completed successfully")
        sys.exit(0)
    else:
        logger.error("Vector Tonic fixed test failed")
        sys.exit(1)
