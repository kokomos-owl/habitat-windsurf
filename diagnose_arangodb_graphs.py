#!/usr/bin/env python
"""
ArangoDB Graph Diagnostics and Repair Utility for Habitat Evolution.

This script provides comprehensive diagnostics and repair capabilities for
ArangoDB graph structures used in the Habitat Evolution system. It helps
identify and fix issues related to the 'edge_collection' error and other
graph initialization problems.

Usage:
    python diagnose_arangodb_graphs.py [--repair] [--force]

Options:
    --repair    Attempt to repair identified issues
    --force     Force recreation of graph structures (use with caution)
"""

import logging
import argparse
import json
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the package
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

class ArangoDBGraphDiagnostics:
    """
    Diagnostic and repair utility for ArangoDB graph structures in Habitat Evolution.
    
    This class provides methods to:
    1. Diagnose issues with graph structures
    2. Verify edge collection configurations
    3. Repair common issues like the 'edge_collection' error
    4. Generate detailed reports for POC to MVP transition
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize the diagnostics utility.
        
        Args:
            db_config: Configuration for the ArangoDB connection
        """
        self.db_config = db_config
        self.connection = None
        self.db = None
        self.issues = []
        self.repair_actions = []
        
    def connect(self) -> bool:
        """
        Connect to the ArangoDB database.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            self.connection = ArangoDBConnection(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 8529),
                username=self.db_config.get("username", "root"),
                password=self.db_config.get("password", ""),
                database_name=self.db_config.get("database_name", "habitat_evolution")
            )
            self.connection.initialize()
            self.db = self.connection._db
            logger.info(f"Connected to ArangoDB: {self.db_config.get('host')}:{self.db_config.get('port')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            self.issues.append({
                "type": "connection_error",
                "description": f"Failed to connect to ArangoDB: {e}",
                "severity": "critical",
                "timestamp": datetime.now().isoformat()
            })
            return False
            
    def diagnose_collections(self) -> Dict[str, Any]:
        """
        Diagnose issues with collections.
        
        Returns:
            Dict[str, Any]: Diagnostic information about collections
        """
        results = {
            "collections": {},
            "issues": []
        }
        
        try:
            # Get all collections
            collections = self.db.collections()
            
            # Check for required collections
            required_collections = ["patterns", "pattern_relationships", "pkm_files"]
            for coll_name in required_collections:
                if not self.db.has_collection(coll_name):
                    issue = {
                        "type": "missing_collection",
                        "collection": coll_name,
                        "description": f"Required collection '{coll_name}' is missing",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat()
                    }
                    results["issues"].append(issue)
                    self.issues.append(issue)
                else:
                    # Get collection details
                    collection = self.db.collection(coll_name)
                    properties = collection.properties()
                    
                    # Check if edge collections are properly configured
                    if coll_name == "pattern_relationships":
                        if properties.get("type") != 3:  # 3 is the type for edge collections
                            issue = {
                                "type": "invalid_edge_collection",
                                "collection": coll_name,
                                "description": f"Collection '{coll_name}' is not configured as an edge collection",
                                "severity": "high",
                                "current_type": properties.get("type"),
                                "timestamp": datetime.now().isoformat()
                            }
                            results["issues"].append(issue)
                            self.issues.append(issue)
                    
                    # Store collection details
                    results["collections"][coll_name] = {
                        "name": coll_name,
                        "type": "edge" if properties.get("type") == 3 else "document",
                        "count": collection.count(),
                        "properties": properties
                    }
            
            logger.info(f"Collection diagnosis complete: {len(results['collections'])} collections found, {len(results['issues'])} issues detected")
            return results
        except Exception as e:
            logger.error(f"Error diagnosing collections: {e}")
            issue = {
                "type": "diagnosis_error",
                "description": f"Error diagnosing collections: {e}",
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            }
            results["issues"].append(issue)
            self.issues.append(issue)
            return results
            
    def diagnose_graphs(self) -> Dict[str, Any]:
        """
        Diagnose issues with graph structures.
        
        Returns:
            Dict[str, Any]: Diagnostic information about graphs
        """
        results = {
            "graphs": {},
            "issues": []
        }
        
        try:
            # Check if pattern_graph exists
            if not self.db.has_graph("pattern_graph"):
                issue = {
                    "type": "missing_graph",
                    "graph": "pattern_graph",
                    "description": "Required graph 'pattern_graph' is missing",
                    "severity": "high",
                    "timestamp": datetime.now().isoformat()
                }
                results["issues"].append(issue)
                self.issues.append(issue)
            else:
                # Get graph details
                graph = self.db.graph("pattern_graph")
                
                try:
                    # Try to access edge definitions - this is where 'edge_collection' error often occurs
                    edge_definitions = graph.edge_definitions()
                    
                    # Check if edge definitions are properly configured
                    pattern_rel_found = False
                    for edge_def in edge_definitions:
                        if edge_def.get("collection") == "pattern_relationships":
                            pattern_rel_found = True
                            
                            # Check from/to collections
                            if "patterns" not in edge_def.get("from", []):
                                issue = {
                                    "type": "invalid_edge_definition",
                                    "graph": "pattern_graph",
                                    "edge_collection": "pattern_relationships",
                                    "description": "Edge definition missing 'patterns' in 'from' collections",
                                    "severity": "high",
                                    "current_definition": edge_def,
                                    "timestamp": datetime.now().isoformat()
                                }
                                results["issues"].append(issue)
                                self.issues.append(issue)
                                
                            if "patterns" not in edge_def.get("to", []):
                                issue = {
                                    "type": "invalid_edge_definition",
                                    "graph": "pattern_graph",
                                    "edge_collection": "pattern_relationships",
                                    "description": "Edge definition missing 'patterns' in 'to' collections",
                                    "severity": "high",
                                    "current_definition": edge_def,
                                    "timestamp": datetime.now().isoformat()
                                }
                                results["issues"].append(issue)
                                self.issues.append(issue)
                    
                    if not pattern_rel_found:
                        issue = {
                            "type": "missing_edge_definition",
                            "graph": "pattern_graph",
                            "description": "Graph missing edge definition for 'pattern_relationships'",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat()
                        }
                        results["issues"].append(issue)
                        self.issues.append(issue)
                        
                    # Store graph details
                    results["graphs"]["pattern_graph"] = {
                        "name": "pattern_graph",
                        "edge_definitions": edge_definitions,
                        "vertex_collections": graph.vertex_collections()
                    }
                except KeyError as ke:
                    # This is likely the 'edge_collection' error
                    if "edge_collection" in str(ke):
                        issue = {
                            "type": "edge_collection_error",
                            "graph": "pattern_graph",
                            "description": "KeyError: 'edge_collection' when accessing graph edge definitions",
                            "severity": "critical",
                            "error": str(ke),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["issues"].append(issue)
                        self.issues.append(issue)
                    else:
                        raise ke
                except Exception as e:
                    issue = {
                        "type": "graph_access_error",
                        "graph": "pattern_graph",
                        "description": f"Error accessing graph details: {e}",
                        "severity": "high",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    results["issues"].append(issue)
                    self.issues.append(issue)
            
            logger.info(f"Graph diagnosis complete: {len(results['graphs'])} graphs found, {len(results['issues'])} issues detected")
            return results
        except Exception as e:
            logger.error(f"Error diagnosing graphs: {e}")
            issue = {
                "type": "diagnosis_error",
                "description": f"Error diagnosing graphs: {e}",
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            }
            results["issues"].append(issue)
            self.issues.append(issue)
            return results
    
    def repair_issues(self, force: bool = False) -> Dict[str, Any]:
        """
        Attempt to repair identified issues.
        
        Args:
            force: If True, force recreation of structures even if it means data loss
            
        Returns:
            Dict[str, Any]: Results of repair operations
        """
        results = {
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "repairs_failed": 0,
            "actions": []
        }
        
        try:
            # Process issues by type
            for issue in self.issues:
                issue_type = issue.get("type")
                
                # Handle missing collections
                if issue_type == "missing_collection":
                    collection_name = issue.get("collection")
                    try:
                        if collection_name == "pattern_relationships":
                            # Create as edge collection
                            self.db.create_collection(collection_name, edge=True)
                            action = {
                                "type": "create_edge_collection",
                                "collection": collection_name,
                                "status": "success",
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            # Create as document collection
                            self.db.create_collection(collection_name)
                            action = {
                                "type": "create_collection",
                                "collection": collection_name,
                                "status": "success",
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        results["repairs_successful"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.info(f"Created collection: {collection_name}")
                    except Exception as e:
                        action = {
                            "type": "create_collection",
                            "collection": collection_name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["repairs_failed"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.error(f"Failed to create collection {collection_name}: {e}")
                
                # Handle invalid edge collections
                elif issue_type == "invalid_edge_collection":
                    collection_name = issue.get("collection")
                    try:
                        if force:
                            # Delete and recreate collection
                            self.db.delete_collection(collection_name)
                            self.db.create_collection(collection_name, edge=True)
                            action = {
                                "type": "recreate_edge_collection",
                                "collection": collection_name,
                                "status": "success",
                                "note": "Collection was deleted and recreated as edge collection",
                                "timestamp": datetime.now().isoformat()
                            }
                            results["repairs_successful"] += 1
                        else:
                            action = {
                                "type": "recreate_edge_collection",
                                "collection": collection_name,
                                "status": "skipped",
                                "note": "Collection needs to be recreated as edge collection, but --force not specified",
                                "timestamp": datetime.now().isoformat()
                            }
                            results["repairs_failed"] += 1
                        
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.info(f"Edge collection fix for {collection_name}: {action['status']}")
                    except Exception as e:
                        action = {
                            "type": "recreate_edge_collection",
                            "collection": collection_name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["repairs_failed"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.error(f"Failed to fix edge collection {collection_name}: {e}")
                
                # Handle missing graphs
                elif issue_type == "missing_graph":
                    graph_name = issue.get("graph")
                    try:
                        # Create graph with proper edge definitions
                        edge_definitions = [
                            {
                                "collection": "pattern_relationships",
                                "from": ["patterns"],
                                "to": ["patterns"]
                            }
                        ]
                        
                        # Ensure edge collection exists first
                        if not self.db.has_collection("pattern_relationships"):
                            self.db.create_collection("pattern_relationships", edge=True)
                            logger.info("Created pattern_relationships edge collection")
                            
                        # Create graph
                        self.db.create_graph(graph_name, edge_definitions)
                        
                        action = {
                            "type": "create_graph",
                            "graph": graph_name,
                            "status": "success",
                            "edge_definitions": edge_definitions,
                            "timestamp": datetime.now().isoformat()
                        }
                        results["repairs_successful"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.info(f"Created graph: {graph_name}")
                    except Exception as e:
                        action = {
                            "type": "create_graph",
                            "graph": graph_name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["repairs_failed"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.error(f"Failed to create graph {graph_name}: {e}")
                
                # Handle edge_collection error
                elif issue_type == "edge_collection_error":
                    graph_name = issue.get("graph")
                    try:
                        if force:
                            # Delete and recreate graph
                            if self.db.has_graph(graph_name):
                                self.db.delete_graph(graph_name)
                                logger.info(f"Deleted graph: {graph_name}")
                            
                            # Ensure edge collection exists
                            if not self.db.has_collection("pattern_relationships"):
                                self.db.create_collection("pattern_relationships", edge=True)
                            elif self.db.collection("pattern_relationships").properties().get("type") != 3:
                                # Delete and recreate as edge collection
                                self.db.delete_collection("pattern_relationships")
                                self.db.create_collection("pattern_relationships", edge=True)
                                
                            # Create graph with proper edge definitions
                            edge_definitions = [
                                {
                                    "collection": "pattern_relationships",
                                    "from": ["patterns"],
                                    "to": ["patterns"]
                                }
                            ]
                            self.db.create_graph(graph_name, edge_definitions)
                            
                            action = {
                                "type": "fix_edge_collection_error",
                                "graph": graph_name,
                                "status": "success",
                                "note": "Graph was deleted and recreated with proper edge definitions",
                                "timestamp": datetime.now().isoformat()
                            }
                            results["repairs_successful"] += 1
                        else:
                            action = {
                                "type": "fix_edge_collection_error",
                                "graph": graph_name,
                                "status": "skipped",
                                "note": "Graph needs to be recreated, but --force not specified",
                                "timestamp": datetime.now().isoformat()
                            }
                            results["repairs_failed"] += 1
                        
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.info(f"Edge collection error fix for {graph_name}: {action['status']}")
                    except Exception as e:
                        action = {
                            "type": "fix_edge_collection_error",
                            "graph": graph_name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["repairs_failed"] += 1
                        results["actions"].append(action)
                        self.repair_actions.append(action)
                        logger.error(f"Failed to fix edge collection error for {graph_name}: {e}")
                
                results["repairs_attempted"] += 1
            
            logger.info(f"Repair operations complete: {results['repairs_successful']} successful, {results['repairs_failed']} failed")
            return results
        except Exception as e:
            logger.error(f"Error during repair operations: {e}")
            return {
                "repairs_attempted": results["repairs_attempted"],
                "repairs_successful": results["repairs_successful"],
                "repairs_failed": results["repairs_failed"] + 1,
                "actions": results["actions"],
                "error": str(e)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic report.
        
        Returns:
            Dict[str, Any]: Diagnostic report
        """
        # Get database version information
        try:
            version_info = self.db.version()
        except:
            version_info = "Unknown"
            
        report = {
            "timestamp": datetime.now().isoformat(),
            "database": {
                "host": self.db_config.get("host"),
                "port": self.db_config.get("port"),
                "database": self.db_config.get("database_name"),
                "version": version_info
            },
            "summary": {
                "total_issues": len(self.issues),
                "critical_issues": len([i for i in self.issues if i.get("severity") == "critical"]),
                "high_issues": len([i for i in self.issues if i.get("severity") == "high"]),
                "repair_actions": len(self.repair_actions),
                "successful_repairs": len([a for a in self.repair_actions if a.get("status") == "success"])
            },
            "issues": self.issues,
            "repair_actions": self.repair_actions,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on diagnostic results.
        
        Returns:
            List[Dict[str, Any]]: List of recommendations
        """
        recommendations = []
        
        # Check for edge_collection error
        if any(i.get("type") == "edge_collection_error" for i in self.issues):
            recommendations.append({
                "priority": "critical",
                "issue": "edge_collection error in ArangoDB graph operations",
                "recommendation": "Run this script with --repair --force to recreate the graph structure",
                "mvp_impact": "This error prevents proper graph operations and must be fixed for MVP",
                "notes": "This will delete and recreate the graph structure, which may result in loss of relationship data"
            })
        
        # Check for invalid edge collections
        if any(i.get("type") == "invalid_edge_collection" for i in self.issues):
            recommendations.append({
                "priority": "high",
                "issue": "Edge collection not properly configured",
                "recommendation": "Run this script with --repair --force to recreate the edge collection",
                "mvp_impact": "Improper edge collection configuration prevents relationship storage",
                "notes": "This will delete and recreate the edge collection, which may result in loss of relationship data"
            })
        
        # Check for missing collections
        if any(i.get("type") == "missing_collection" for i in self.issues):
            recommendations.append({
                "priority": "high",
                "issue": "Missing required collections",
                "recommendation": "Run this script with --repair to create missing collections",
                "mvp_impact": "Missing collections prevent proper data storage",
                "notes": "This will create the missing collections without data loss"
            })
        
        # Add general recommendations
        if len(self.issues) > 0:
            recommendations.append({
                "priority": "medium",
                "issue": "ArangoDB structure issues detected",
                "recommendation": "Review ArangoDB Python driver version and compatibility",
                "mvp_impact": "Ensures long-term stability of graph operations",
                "notes": "Consider upgrading to the latest compatible version of the ArangoDB Python driver"
            })
            
            recommendations.append({
                "priority": "medium",
                "issue": "Potential data inconsistency due to fallback mechanisms",
                "recommendation": "Implement a data reconciliation process to merge fallback data with database",
                "mvp_impact": "Ensures data integrity across system restarts",
                "notes": "This should be implemented before moving to MVP"
            })
        
        return recommendations

def main():
    """Run the ArangoDB graph diagnostics and repair utility."""
    parser = argparse.ArgumentParser(description="ArangoDB Graph Diagnostics and Repair Utility")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair identified issues")
    parser.add_argument("--force", action="store_true", help="Force recreation of graph structures (use with caution)")
    args = parser.parse_args()
    
    logger.info("Starting ArangoDB Graph Diagnostics")
    
    # Database configuration
    db_config = {
        "host": "localhost",
        "port": 8529,
        "username": "root",
        "password": "habitat",
        "database_name": "habitat_evolution"
    }
    
    # Create diagnostics utility
    diagnostics = ArangoDBGraphDiagnostics(db_config)
    
    # Connect to database
    if not diagnostics.connect():
        logger.error("Failed to connect to database, exiting")
        sys.exit(1)
    
    # Run diagnostics
    collection_results = diagnostics.diagnose_collections()
    graph_results = diagnostics.diagnose_graphs()
    
    # Generate report
    report = diagnostics.generate_report()
    
    # Print summary
    print("\n=== ArangoDB Graph Diagnostics Summary ===")
    print(f"Total issues found: {report['summary']['total_issues']}")
    print(f"  Critical issues: {report['summary']['critical_issues']}")
    print(f"  High priority issues: {report['summary']['high_issues']}")
    
    # Print issues
    if report['summary']['total_issues'] > 0:
        print("\n=== Issues ===")
        for i, issue in enumerate(report['issues']):
            print(f"{i+1}. [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")
    
    # Print recommendations
    if report['recommendations']:
        print("\n=== Recommendations ===")
        for i, rec in enumerate(report['recommendations']):
            print(f"{i+1}. [{rec['priority'].upper()}] {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   MVP Impact: {rec['mvp_impact']}")
    
    # Attempt repairs if requested
    if args.repair:
        print("\n=== Attempting Repairs ===")
        repair_results = diagnostics.repair_issues(force=args.force)
        
        print(f"Repairs attempted: {repair_results['repairs_attempted']}")
        print(f"Repairs successful: {repair_results['repairs_successful']}")
        print(f"Repairs failed: {repair_results['repairs_failed']}")
        
        # Run diagnostics again to verify repairs
        print("\n=== Post-Repair Diagnostics ===")
        collection_results = diagnostics.diagnose_collections()
        graph_results = diagnostics.diagnose_graphs()
        
        # Generate updated report
        report = diagnostics.generate_report()
        
        print(f"Remaining issues: {report['summary']['total_issues']}")
    
    # Save report to file
    report_file = f"arangodb_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with error code if issues remain
    if report['summary']['total_issues'] > 0:
        sys.exit(1)
    else:
        print("\nNo issues found or all issues successfully repaired.")
        sys.exit(0)

if __name__ == "__main__":
    main()
