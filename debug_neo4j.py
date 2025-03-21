#!/usr/bin/env python3
"""
Script to debug Neo4j connection and parameter issues
"""
import os
import sys
from neo4j import GraphDatabase

# Neo4j configuration
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "habitat123"

def test_neo4j_connection():
    """Test Neo4j connection and parameter handling"""
    print(f"Connecting to Neo4j at {neo4j_uri} with user {neo4j_user}")
    
    try:
        # Create Neo4j driver
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j' as message")
            print(result.single()["message"])
        
        # Test parameter handling
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create a test domain with explicit parameters
            params = {
                "id": "test-domain",
                "name": "Test Domain",
                "dominant_frequency": 0.5,
                "bandwidth": 0.1,
                "phase_coherence": 0.8,
                "radius": 0.7
            }
            
            print(f"Creating test domain with parameters: {params}")
            
            # Create domain
            session.run(
                """
                CREATE (fd:FrequencyDomain {
                    id: $id, 
                    name: $name, 
                    dominant_frequency: $dominant_frequency, 
                    bandwidth: $bandwidth, 
                    phase_coherence: $phase_coherence, 
                    radius: $radius
                })
                """,
                params
            )
            
            # Verify domain was created
            result = session.run("MATCH (fd:FrequencyDomain) RETURN fd")
            records = result.data()
            print(f"Created {len(records)} domain(s):")
            for record in records:
                print(f"  {record}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_neo4j_connection()
