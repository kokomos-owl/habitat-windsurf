#!/usr/bin/env python
"""
Habitat Evolution Diagnostic Runner

This script runs comprehensive diagnostic tests on the Habitat Evolution system
to identify initialization issues, dependency problems, and component failures.
It uses enhanced logging and verification to provide detailed error tracking.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Configure argument parser
parser = argparse.ArgumentParser(description='Run Habitat Evolution diagnostics')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
parser.add_argument('--log-dir', default='logs', help='Directory to store log files')
parser.add_argument('--report-dir', default='diagnostic_reports', help='Directory to store diagnostic reports')
parser.add_argument('--test-file', default='tests/diagnostic/test_component_diagnostics.py', 
                    help='Path to the diagnostic test file')
parser.add_argument('--test-name', default=None, 
                    help='Specific test to run (e.g., test_component_diagnostic)')
args = parser.parse_args()

# Create directories if they don't exist
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.report_dir, exist_ok=True)

# Generate timestamp for this diagnostic run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"diagnostic_run_{timestamp}"
run_dir = os.path.join(args.report_dir, run_id)
os.makedirs(run_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(args.log_dir, f"{run_id}.log")
log_level = logging.DEBUG if args.verbose else logging.INFO

# Configure root logger
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("diagnostic_runner")
logger.info(f"Starting Habitat Evolution diagnostic run: {run_id}")
logger.info(f"Log file: {log_file}")
logger.info(f"Report directory: {run_dir}")

# Run the diagnostic tests
import pytest

logger.info("Running diagnostic tests...")

# Construct the pytest command
pytest_args = ["-v"]
if args.test_name:
    pytest_args.append(f"{args.test_file}::{args.test_name}")
else:
    pytest_args.append(args.test_file)

# We'll capture the output directly rather than using the json-report plugin

# Run the tests
logger.info(f"Running pytest with args: {pytest_args}")
pytest_exit_code = pytest.main(pytest_args)

# Collect and organize all diagnostic reports
logger.info("Collecting diagnostic reports...")

report_files = [
    "component_diagnostic_report.json",
    "dependency_analysis.json",
    "method_tracing_results.json"
]

consolidated_report = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "pytest_exit_code": pytest_exit_code,
    "reports": {}
}

for report_file in report_files:
    try:
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report_data = json.load(f)
                consolidated_report["reports"][os.path.splitext(report_file)[0]] = report_data
            
            # Move the report file to the run directory
            os.rename(report_file, os.path.join(run_dir, report_file))
            logger.info(f"Collected report: {report_file}")
        else:
            logger.warning(f"Report file not found: {report_file}")
    except Exception as e:
        logger.error(f"Error processing report file {report_file}: {e}")

# Save consolidated report
consolidated_report_file = os.path.join(run_dir, "consolidated_report.json")
with open(consolidated_report_file, 'w') as f:
    json.dump(consolidated_report, f, indent=2)

logger.info(f"Consolidated report saved to: {consolidated_report_file}")

# Generate a summary of the diagnostic run
summary = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "test_status": "PASSED" if pytest_exit_code == 0 else "FAILED",
    "component_issues": []
}

# Extract component issues from the diagnostic report
if "component_diagnostic_report" in consolidated_report["reports"]:
    diagnostic_report = consolidated_report["reports"]["component_diagnostic_report"]
    
    # Check for initialization errors
    if "initialization_errors" in diagnostic_report:
        for component, error in diagnostic_report["initialization_errors"].items():
            summary["component_issues"].append({
                "component": component,
                "issue_type": "initialization",
                "error": error
            })
    
    # Check verification results
    if "verification_results" in diagnostic_report:
        for component, result in diagnostic_report["verification_results"].items():
            if not result.get("verification_passed", False):
                errors = result.get("errors", [])
                summary["component_issues"].append({
                    "component": component,
                    "issue_type": "verification",
                    "errors": errors
                })

# Save summary
summary_file = os.path.join(run_dir, "summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary to console
logger.info("\n" + "="*50)
logger.info(f"DIAGNOSTIC RUN SUMMARY: {run_id}")
logger.info(f"Status: {summary['test_status']}")
logger.info(f"Component Issues: {len(summary['component_issues'])}")

for issue in summary["component_issues"]:
    logger.info(f"  - {issue['component']}: {issue['issue_type']} issue")
    if "error" in issue:
        logger.info(f"    Error: {issue['error']}")
    elif "errors" in issue:
        for error in issue["errors"]:
            logger.info(f"    Error: {error}")

logger.info("="*50)
logger.info(f"Diagnostic reports saved to: {run_dir}")
logger.info(f"Log file: {log_file}")

# Exit with the pytest exit code
sys.exit(pytest_exit_code)
