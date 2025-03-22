#!/usr/bin/env python
"""
Initialize ArangoDB for Habitat
This script creates the necessary database and collections for Habitat's domain-predicate tracking.
"""

from arango import ArangoClient

def init_database():
    """Initialize the ArangoDB database structure for Habitat."""
    # Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    sys_db = client.db("_system", username="root", password="habitat")
    
    # Create habitat database if it doesn't exist
    if not sys_db.has_database("habitat"):
        sys_db.create_database("habitat")
        print("Created 'habitat' database")
    else:
        print("'habitat' database already exists")
    
    # Connect to habitat database
    db = client.db("habitat", username="root", password="habitat")
    
    # Create collections
    collections = [
        # Document collections
        ("Document", False, "Documents processed by Habitat"),
        ("Domain", False, "Semantic domains identified within documents"),
        ("Predicate", False, "Subject-verb-object structures extracted from text"),
        ("Actant", False, "Entities that appear as subjects or objects in predicates"),
        ("SemanticTheme", False, "Themes extracted from semantic domains"),
        
        # Edge collections
        ("DocumentContainsDomain", True, "Links documents to their domains"),
        ("DomainContainsPredicate", True, "Links domains to predicates"),
        ("PredicateHasSubject", True, "Links predicates to subject actants"),
        ("PredicateHasObject", True, "Links predicates to object actants"),
        ("DomainHasTheme", True, "Links domains to semantic themes"),
        ("ActantAppearsIn", True, "Tracks actant appearances across domains"),
        ("DomainResonatesWith", True, "Connects resonating domains across documents")
    ]
    
    for name, is_edge, description in collections:
        if not db.has_collection(name):
            db.create_collection(name, edge=is_edge)
            # Store description as a property in the first document
            collection = db.collection(name)
            # We'll store collection metadata in a separate metadata collection
            print(f"Created collection: {name}")
        else:
            print(f"Collection already exists: {name}")
    
    # Create indexes for faster lookups
    if "Document" in db.collections():
        db.collection("Document").add_hash_index(["doc_id"], unique=True)
        print("Added index on Document.doc_id")
    
    if "Domain" in db.collections():
        db.collection("Domain").add_hash_index(["domain_id"], unique=True)
        print("Added index on Domain.domain_id")
    
    if "Predicate" in db.collections():
        db.collection("Predicate").add_hash_index(["predicate_id"], unique=True)
        print("Added index on Predicate.predicate_id")
    
    if "Actant" in db.collections():
        db.collection("Actant").add_hash_index(["name"], unique=True)
        print("Added index on Actant.name")
    
    if "SemanticTheme" in db.collections():
        db.collection("SemanticTheme").add_hash_index(["text"], unique=True)
        print("Added index on SemanticTheme.text")
    
    print("\nDatabase initialization complete")
    print("You can now access the ArangoDB web interface at http://localhost:8529")
    print("Username: root, Password: habitat")
    print("Database: habitat")

if __name__ == "__main__":
    init_database()
