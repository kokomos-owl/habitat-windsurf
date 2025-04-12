#!/usr/bin/env python
"""
Test script to verify the create_document method implementation in ArangoDBConnection.
"""

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
import uuid

def test_arangodb_document_methods():
    """Test and compare document creation methods in ArangoDBConnection."""
    try:
        # Initialize connection with credentials
        conn = ArangoDBConnection(
            username="root",
            password="habitat",
            database_name="habitat_evolution_test"
        )
        conn.initialize()
        print('ArangoDBConnection initialized')
        
        # Check if create_document method exists (our custom method)
        if hasattr(conn, 'create_document'):
            print('✅ create_document method exists')
        else:
            print('❌ create_document method does NOT exist')
            
        # Check if insert method exists (standard ArangoDB method)
        if hasattr(conn, 'insert'):
            print('✅ insert method exists')
        else:
            print('❌ insert method does NOT exist')
        
        # Create a test collection if needed
        collection_name = 'test_collection'
        if not conn._db.has_collection(collection_name):
            conn.create_collection(collection_name)
            print(f'Created {collection_name}')
        
        # Test 1: Using insert method (ArangoDB standard)
        try:
            # Create a test document with insert
            insert_doc = {
                'test_id': f'insert-{uuid.uuid4()}',
                'name': 'Insert Test Document',
                'description': 'Testing insert method (ArangoDB standard)',
                'method': 'insert'
            }
            
            # Use the insert method
            insert_result = conn.insert(collection_name, insert_doc)
            print(f'✅ Document created with insert(): {insert_result["_id"] if "_id" in insert_result else "ID not found"}')
            
            # Verify document was created
            if '_id' in insert_result:
                key = insert_result['_id'].split('/')[1]
                retrieved = conn.get_document(collection_name, key)
                if retrieved:
                    print(f'✅ insert() document retrieved: {retrieved["name"]}')
                else:
                    print('❌ Failed to retrieve insert() document')
        except Exception as e:
            print(f'❌ insert() error: {e}')
        
        # Test 2: Using create_document method (our custom method)
        try:
            # Create a test document with create_document
            create_doc = {
                'test_id': f'create-{uuid.uuid4()}',
                'name': 'Create Document Test',
                'description': 'Testing create_document method implementation',
                'method': 'create_document'
            }
            
            # Use the create_document method
            create_result = conn.create_document(collection_name, create_doc)
            print(f'✅ Document created with create_document(): {create_result["_id"] if "_id" in create_result else "ID not found"}')
            
            # Verify document was created
            if '_id' in create_result:
                key = create_result['_id'].split('/')[1]
                retrieved = conn.get_document(collection_name, key)
                if retrieved:
                    print(f'✅ create_document() document retrieved: {retrieved["name"]}')
                else:
                    print('❌ Failed to retrieve create_document() document')
        except Exception as e:
            print(f'❌ create_document() error: {e}')
            
        # Test 3: Collection creation behavior
        try:
            # Test with a non-existent collection
            new_collection = f'test_collection_{uuid.uuid4().hex[:8]}'
            
            # Try create_document on non-existent collection (should create it)
            auto_doc = {
                'test_id': f'auto-{uuid.uuid4()}',
                'name': 'Auto-Created Collection Test',
                'method': 'create_document_auto_collection'
            }
            
            # This should create the collection automatically
            auto_result = conn.create_document(new_collection, auto_doc)
            print(f'✅ Document created with auto-collection creation: {auto_result["_id"] if "_id" in auto_result else "ID not found"}')
            
            # Verify collection was created
            if conn._db.has_collection(new_collection):
                print(f'✅ Collection {new_collection} was automatically created')
            else:
                print(f'❌ Collection {new_collection} was NOT created')
                
        except Exception as e:
            print(f'❌ Auto-collection test error: {e}')
            
    except Exception as e:
        print(f'❌ Connection error: {e}')

if __name__ == "__main__":
    test_arangodb_document_methods()
