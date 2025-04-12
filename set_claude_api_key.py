#!/usr/bin/env python
"""
Script to set the Claude API key in the environment.

This script prompts the user for their Claude API key and sets it in the environment
for the current session. It also provides an option to save the key to a local
configuration file for future use.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def set_claude_api_key():
    """
    Set the Claude API key in the environment.
    
    This function prompts the user for their Claude API key and sets it in the
    environment for the current session. It also provides an option to save the
    key to a local configuration file for future use.
    
    Returns:
        bool: True if the key was set successfully, False otherwise
    """
    try:
        # Check if the key is already set
        existing_key = os.environ.get("CLAUDE_API_KEY")
        if existing_key:
            logger.info("Claude API key is already set in the environment")
            return True
        
        # Check if the key is saved in the config file
        config_dir = Path.home() / ".habitat_alpha"
        config_file = config_dir / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    if "claude_api_key" in config and config["claude_api_key"]:
                        os.environ["CLAUDE_API_KEY"] = config["claude_api_key"]
                        logger.info("Claude API key loaded from config file")
                        return True
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        # Prompt the user for the key
        print("\n=== Claude API Key Setup ===")
        print("The Claude API key is required for the Habitat Evolution system to function properly.")
        print("You can find your Claude API key in the Anthropic Console: https://console.anthropic.com/")
        print("\nThe key will be used for this session only and will not be stored permanently unless you choose to do so.")
        
        api_key = input("\nEnter your Claude API key (starts with 'sk-ant-'): ")
        
        if not api_key:
            logger.error("No API key provided")
            return False
        
        if not api_key.startswith("sk-ant-"):
            logger.warning("The provided key doesn't start with 'sk-ant-', which is the expected format for Claude API keys")
            confirm = input("Are you sure this is the correct key? (y/n): ")
            if confirm.lower() != "y":
                logger.error("API key setup canceled")
                return False
        
        # Set the key in the environment
        os.environ["CLAUDE_API_KEY"] = api_key
        logger.info("Claude API key set in environment")
        
        # Ask if the user wants to save the key
        save_key = input("\nWould you like to save this key for future use? (y/n): ")
        if save_key.lower() == "y":
            # Create config directory if it doesn't exist
            config_dir.mkdir(exist_ok=True)
            
            # Save the key to the config file
            config = {}
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                except Exception:
                    pass
            
            config["claude_api_key"] = api_key
            
            with open(config_file, "w") as f:
                json.dump(config, f)
            
            logger.info(f"Claude API key saved to {config_file}")
        
        print("\nClaude API key set successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error setting Claude API key: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Claude API key setup")
    success = set_claude_api_key()
    
    if success:
        logger.info("Claude API key setup completed successfully")
        sys.exit(0)
    else:
        logger.error("Claude API key setup failed")
        sys.exit(1)
