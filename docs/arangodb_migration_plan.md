# Habitat Migration Plan: Neo4j to ArangoDB

## Overview

This document outlines the plan for migrating Habitat from Neo4j to ArangoDB to better support our unique needs in tracking domain-predicates, concept relationships, and evolutionary patterns. The migration will be implemented using Docker containers for easier deployment and management.

## 1. Docker Setup for ArangoDB

### 1.1 Docker Compose Configuration

Create a `docker-compose.yml` file in the project root:

```yaml
version: '3.8'

services:
  arangodb:
    image: arangodb:3.10.4
    environment:
      - ARANGO_ROOT_PASSWORD=habitat
    ports:
      - "8529:8529"
    volumes:
      - arangodb_data:/var/lib/arangodb3
      - arangodb_apps:/var/lib/arangodb3-apps
    restart: unless-stopped

volumes:
  arangodb_data:
  arangodb_apps:
```

### 1.2 Starting the ArangoDB Container

```bash
docker-compose up -d
```

Access the ArangoDB web interface at [http://localhost:8529](http://localhost:8529) with username `root` and password `habitat`.

### 1.3 Database Initialization

Create a Python script to initialize the database structure:

```python
# scripts/init_arangodb.py
from arango import ArangoClient

def init_database():
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    sys_db = client.db("_system", username="root", password="habitat")
    
    # Create habitat database if it doesn't exist
    if not sys_db.has_database("habitat"):
        sys_db.create_database("habitat")
    
    # Connect to habitat database
    db = client.db("habitat", username="root", password="habitat")
    
    # Create collections
    collections = [
        # Document collections
        ("Document", False),
        ("Domain", False),
        ("Predicate", False),
        ("Actant", False),
        ("SemanticTheme", False),
        
        # Edge collections
        ("DocumentContainsDomain", True),
        ("DomainContainsPredicate", True),
        ("PredicateHasSubject", True),
        ("PredicateHasObject", True),
        ("DomainHasTheme", True),
        ("ActantAppearsIn", True),
        ("DomainResonatesWith", True)
    ]
    
    for name, is_edge in collections:
        if not db.has_collection(name):
            db.create_collection(name, edge=is_edge)
    
    # Create indexes
    db.collection("Document").add_hash_index(["doc_id"], unique=True)
    db.collection("Domain").add_hash_index(["domain_id"], unique=True)
    db.collection("Predicate").add_hash_index(["predicate_id"], unique=True)
    db.collection("Actant").add_hash_index(["name"], unique=True)
    
    print("Database initialization complete")

if __name__ == "__main__":
    init_database()
```

Run the initialization script:

```bash
python scripts/init_arangodb.py
```

## 2. Data Model Migration

### 2.1 Data Model Mapping

| Neo4j Model | ArangoDB Model |
|-------------|----------------|
| Nodes with labels | Document collections |
| Relationships | Edge collections |
| Node properties | Document attributes |
| Relationship properties | Edge attributes |

### 2.2 Collection Structure

- **Document Collections:**
  - `Document`: Stores document metadata
  - `Domain`: Semantic domains identified within documents
  - `Predicate`: Subject-verb-object structures
  - `Actant`: Entities that appear as subjects or objects
  - `SemanticTheme`: Themes extracted from domains

- **Edge Collections:**
  - `DocumentContainsDomain`: Links documents to their domains
  - `DomainContainsPredicate`: Links domains to predicates
  - `PredicateHasSubject`: Links predicates to subject actants
  - `PredicateHasObject`: Links predicates to object actants
  - `DomainHasTheme`: Links domains to semantic themes
  - `ActantAppearsIn`: Tracks actant appearances across domains
  - `DomainResonatesWith`: Connects resonating domains

### 2.3 Migration Script

Create a script to migrate existing Neo4j data to ArangoDB:

```python
# scripts/migrate_neo4j_to_arango.py
from arango import ArangoClient
from neo4j import GraphDatabase

def migrate_data(neo4j_uri, neo4j_user, neo4j_password):
    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri, 
        auth=(neo4j_user, neo4j_password)
    )
    
    # Connect to ArangoDB
    arango_client = ArangoClient(hosts="http://localhost:8529")
    db = arango_client.db("habitat", username="root", password="habitat")
    
    # Migrate documents
    with neo4j_driver.session() as session:
        # Migrate document nodes
        result = session.run("MATCH (d:Document) RETURN d")
        for record in result:
            node = record["d"]
            doc = {
                "doc_id": node.get("id"),
                "title": node.get("title", ""),
                "path": node.get("path", ""),
                "created_at": node.get("created_at", ""),
                "neo4j_id": node.id  # Store Neo4j ID for reference
            }
            db.collection("Document").insert(doc)
        
        # Migrate domain nodes
        # ... similar pattern for other node types
        
        # Migrate relationships
        # ... similar pattern for relationships
    
    neo4j_driver.close()
    print("Migration complete")

if __name__ == "__main__":
    migrate_data(
        "bolt://localhost:7687", 
        "neo4j", 
        "habitat"  # Replace with your Neo4j password
    )
```

## 3. Adapting Code to ArangoDB

### 3.1 Python Driver Installation

```bash
pip install python-arango
```

### 3.2 Database Connection Class

```python
# habitat_evolution/persistence/arango_connector.py
from arango import ArangoClient

class ArangoConnector:
    def __init__(self, host="http://localhost:8529", db_name="habitat", 
                 username="root", password="habitat"):
        self.client = ArangoClient(hosts=host)
        self.db = self.client.db(db_name, username=username, password=password)
    
    def get_collection(self, collection_name):
        return self.db.collection(collection_name)
    
    def get_edge_collection(self, collection_name):
        return self.db.collection(collection_name)
    
    def execute_query(self, query, bind_vars=None):
        return self.db.aql.execute(query, bind_vars=bind_vars)
```

### 3.3 Query Translation Examples

#### Neo4j to ArangoDB Query Translation

| Neo4j (Cypher) | ArangoDB (AQL) |
|----------------|----------------|
| `MATCH (d:Document) RETURN d` | `FOR d IN Document RETURN d` |
| `MATCH (d:Document)-[:CONTAINS]->(dom:Domain) RETURN d, dom` | `FOR d IN Document FOR dom IN OUTBOUND d DocumentContainsDomain RETURN {d, dom}` |
| `MATCH (a)-[r:RELATES_TO]->(b) WHERE r.weight > 0.5 RETURN a, r, b` | `FOR a, r, b IN OUTBOUND 'Actant/123' ActantRelatesTo FILTER r.weight > 0.5 RETURN {a, r, b}` |

## 4. Domain-Predicate Tracking Implementation

### 4.1 Domain Storage

```python
# habitat_evolution/persistence/domain_repository.py
from habitat_evolution.persistence.arango_connector import ArangoConnector

class DomainRepository:
    def __init__(self, connector=None):
        self.connector = connector or ArangoConnector()
        self.domains = self.connector.get_collection("Domain")
        self.doc_contains_domain = self.connector.get_edge_collection("DocumentContainsDomain")
        self.domain_has_theme = self.connector.get_edge_collection("DomainHasTheme")
    
    def store_domain(self, document_key, domain_data):
        """Store a domain and connect it to its document."""
        # Insert domain
        domain_doc = {
            "domain_id": domain_data["id"],
            "start_idx": domain_data["boundary"][0],
            "end_idx": domain_data["boundary"][1],
            "coherence": domain_data.get("coherence", 0)
        }
        domain_meta = self.domains.insert(domain_doc, return_new=True)
        domain_key = domain_meta["_key"]
        
        # Connect domain to document
        self.doc_contains_domain.insert({
            "_from": f"Document/{document_key}",
            "_to": f"Domain/{domain_key}"
        })
        
        # Store themes
        for theme in domain_data.get("themes", []):
            theme_key = self._store_theme(theme)
            self.domain_has_theme.insert({
                "_from": f"Domain/{domain_key}",
                "_to": f"SemanticTheme/{theme_key}"
            })
        
        return domain_key
    
    def _store_theme(self, theme_text):
        """Store a semantic theme."""
        themes = self.connector.get_collection("SemanticTheme")
        
        # Check if theme already exists
        query = """
        FOR t IN SemanticTheme
            FILTER t.text == @text
            RETURN t
        """
        results = list(self.connector.execute_query(query, {"text": theme_text}))
        
        if results:
            return results[0]["_key"]
        
        # Insert new theme
        theme_meta = themes.insert({"text": theme_text}, return_new=True)
        return theme_meta["_key"]
```

### 4.2 Predicate Storage

```python
# habitat_evolution/persistence/predicate_repository.py
from habitat_evolution.persistence.arango_connector import ArangoConnector

class PredicateRepository:
    def __init__(self, connector=None):
        self.connector = connector or ArangoConnector()
        self.predicates = self.connector.get_collection("Predicate")
        self.actants = self.connector.get_collection("Actant")
        self.domain_contains_predicate = self.connector.get_edge_collection("DomainContainsPredicate")
        self.predicate_has_subject = self.connector.get_edge_collection("PredicateHasSubject")
        self.predicate_has_object = self.connector.get_edge_collection("PredicateHasObject")
    
    def store_predicate(self, domain_key, predicate_data):
        """Store a predicate and its relationships."""
        # Insert predicate
        predicate_doc = {
            "predicate_id": predicate_data.get("id"),
            "verb": predicate_data["verb"],
            "sentence": predicate_data.get("sentence", "")
        }
        predicate_meta = self.predicates.insert(predicate_doc, return_new=True)
        predicate_key = predicate_meta["_key"]
        
        # Connect predicate to domain
        self.domain_contains_predicate.insert({
            "_from": f"Domain/{domain_key}",
            "_to": f"Predicate/{predicate_key}"
        })
        
        # Store subject actant
        subject_key = self._store_actant(predicate_data["subject"])
        self.predicate_has_subject.insert({
            "_from": f"Predicate/{predicate_key}",
            "_to": f"Actant/{subject_key}"
        })
        
        # Store object actant
        object_key = self._store_actant(predicate_data["object"])
        self.predicate_has_object.insert({
            "_from": f"Predicate/{predicate_key}",
            "_to": f"Actant/{object_key}"
        })
        
        return predicate_key
    
    def _store_actant(self, actant_name):
        """Store an actant (subject or object entity)."""
        # Check if actant already exists
        query = """
        FOR a IN Actant
            FILTER a.name == @name
            RETURN a
        """
        results = list(self.connector.execute_query(query, {"name": actant_name}))
        
        if results:
            return results[0]["_key"]
        
        # Insert new actant
        actant_meta = self.actants.insert({"name": actant_name}, return_new=True)
        return actant_meta["_key"]
```

### 4.3 Actant Tracking Across Domains

```python
# habitat_evolution/persistence/actant_repository.py
from habitat_evolution.persistence.arango_connector import ArangoConnector

class ActantRepository:
    def __init__(self, connector=None):
        self.connector = connector or ArangoConnector()
        self.actants = self.connector.get_collection("Actant")
        self.actant_appears_in = self.connector.get_edge_collection("ActantAppearsIn")
    
    def track_actant_appearance(self, actant_key, domain_key, role, predicate_key):
        """Track an actant's appearance in a domain."""
        self.actant_appears_in.insert({
            "_from": f"Actant/{actant_key}",
            "_to": f"Domain/{domain_key}",
            "role": role,
            "predicate_key": predicate_key
        })
    
    def get_actant_journey(self, actant_name):
        """Get the journey of an actant across domains."""
        query = """
        FOR a IN Actant
            FILTER a.name == @name
            FOR appearance IN OUTBOUND a ActantAppearsIn
                FOR domain IN Domain
                    FILTER appearance._to == CONCAT('Domain/', domain._key)
                    FOR theme IN OUTBOUND domain DomainHasTheme
                        FOR predicate IN Predicate
                            FILTER predicate._key == appearance.predicate_key
                            RETURN {
                                domain: domain,
                                theme: theme,
                                role: appearance.role,
                                predicate: predicate
                            }
        """
        return list(self.connector.execute_query(query, {"name": actant_name}))
```

## 5. Incremental Migration Strategy

### 5.1 Phase 1: Setup and Parallel Operation

1. **Set up ArangoDB Docker container**
2. **Implement basic ArangoDB repositories**
3. **Create an abstraction layer to support both databases**

```python
# habitat_evolution/persistence/repository_factory.py
class RepositoryFactory:
    def __init__(self, db_type="neo4j"):
        self.db_type = db_type
    
    def create_document_repository(self):
        if self.db_type == "neo4j":
            from habitat_evolution.persistence.neo4j.document_repository import DocumentRepository
            return DocumentRepository()
        else:
            from habitat_evolution.persistence.arango.document_repository import DocumentRepository
            return DocumentRepository()
    
    # Similar methods for other repository types
```

### 5.2 Phase 2: Data Migration and Validation

1. **Migrate core data structures**
2. **Implement validation tests**
3. **Run parallel queries and compare results**

### 5.3 Phase 3: Gradual Feature Transition

1. **Move domain detection to ArangoDB**
2. **Implement predicate extraction in ArangoDB**
3. **Add actant tracking in ArangoDB**

### 5.4 Phase 4: Complete Transition

1. **Move all remaining functionality to ArangoDB**
2. **Remove Neo4j dependencies**
3. **Optimize ArangoDB queries and indexes**

## 6. Testing Strategy

### 6.1 Unit Tests

Create unit tests for each repository class:

```python
# tests/persistence/test_arango_domain_repository.py
import unittest
from habitat_evolution.persistence.domain_repository import DomainRepository

class TestArangoDomainRepository(unittest.TestCase):
    def setUp(self):
        # Set up test database
        self.repo = DomainRepository()
    
    def test_store_domain(self):
        # Test domain storage
        domain_data = {
            "id": "test-domain-1",
            "boundary": (0, 10),
            "coherence": 0.85,
            "themes": ["climate", "risk", "adaptation"]
        }
        domain_key = self.repo.store_domain("test-doc-1", domain_data)
        
        # Verify domain was stored
        domain = self.repo.connector.get_collection("Domain").get(domain_key)
        self.assertEqual(domain["domain_id"], "test-domain-1")
        self.assertEqual(domain["coherence"], 0.85)
        
        # Verify themes were stored
        query = """
        FOR d IN Domain
            FILTER d._key == @key
            FOR theme IN OUTBOUND d DomainHasTheme
                RETURN theme.text
        """
        themes = list(self.repo.connector.execute_query(query, {"key": domain_key}))
        self.assertIn("climate", themes)
        self.assertIn("risk", themes)
        self.assertIn("adaptation", themes)
```

### 6.2 Integration Tests

Create integration tests that verify the entire pipeline:

```python
# tests/integration/test_arango_document_processing.py
import unittest
from habitat_evolution.document_processor import DocumentProcessor

class TestArangoDocumentProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(db_type="arango")
    
    def test_process_document(self):
        # Test processing a document
        doc_path = "tests/data/boston_harbor_islands.txt"
        result = self.processor.process_document(doc_path)
        
        # Verify document was processed
        self.assertIsNotNone(result["doc_id"])
        self.assertGreater(len(result["domains"]), 0)
        
        # Verify domains have predicates
        for domain in result["domains"]:
            self.assertIn("predicates", domain)
            if domain["predicates"]:
                self.assertIn("subject", domain["predicates"][0])
                self.assertIn("verb", domain["predicates"][0])
                self.assertIn("object", domain["predicates"][0])
```

## 7. Performance Considerations

### 7.1 Indexing Strategy

- Create compound indexes for frequently queried fields
- Use hash indexes for exact lookups
- Use persistent indexes for range queries

### 7.2 Query Optimization

- Use AQL query profiling to identify slow queries
- Optimize complex traversals
- Use graph functions for path finding

### 7.3 Scaling Considerations

- Configure ArangoDB for appropriate memory usage
- Set up sharding for larger datasets
- Consider ArangoDB cluster for production deployment

## 8. Conclusion

This migration plan provides a structured approach to transitioning Habitat from Neo4j to ArangoDB. By following this incremental approach, we can minimize disruption while enhancing Habitat's ability to track domain-predicates, concept relationships, and evolutionary patterns.

The ArangoDB implementation will provide a more flexible foundation for Habitat's growth, with better support for schema evolution and improved scaling characteristics as we move from proof-of-concept to a full-scale system.
