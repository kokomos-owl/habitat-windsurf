from neo4j import GraphDatabase

def check_neo4j_data():
    # Connect to Neo4j
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "habitat123"))
    
    try:
        with driver.session() as session:
            # Check pattern nodes
            result = session.run("MATCH (p:Pattern) RETURN count(p) as pattern_count")
            pattern_count = result.single()["pattern_count"]
            print(f"\nPattern count: {pattern_count}")
            
            # Get pattern details
            result = session.run("""
                MATCH (p:Pattern)
                RETURN p {.*} as pattern
                LIMIT 5
            """)
            print("\nPattern details:")
            for record in result:
                print(f"\n{record['pattern']}")
            
            # Check relationships
            result = session.run("""
                MATCH (p1:Pattern)-[r]->(p2:Pattern)
                RETURN type(r) as rel_type, count(r) as rel_count
            """)
            print("\nRelationship counts:")
            for record in result:
                print(f"{record['rel_type']}: {record['rel_count']}")
            
    finally:
        driver.close()

if __name__ == "__main__":
    check_neo4j_data()
