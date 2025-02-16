# Test Environment Setup Guide

## Environment Variable Configuration

Merge these test-specific variables with your existing external configuration:

```bash
# =================================================================
# Pattern-Aware RAG Test Configuration
# =================================================================

# -----------------------------
# Test-Specific Settings
# -----------------------------
TEST_ENVIRONMENT=development
TEST_LOG_LEVEL=DEBUG

# Pattern-Aware RAG Test Directories
TEST_PERSIST_DIR=./.habitat/test_data
TEST_CACHE_DIR=./.habitat/test_cache
TEST_EMBEDDING_DIR=./.habitat/embeddings

# -----------------------------
# Learning Window Configuration
# -----------------------------
TEST_WINDOW_DURATION=5  # minutes
TEST_MAX_CHANGES=10
TEST_STABILITY_THRESHOLD=0.7
TEST_COHERENCE_THRESHOLD=0.6

# -----------------------------
# Back Pressure Configuration
# -----------------------------
TEST_BASE_DELAY=0.1
TEST_MAX_DELAY=2.0
TEST_PRESSURE_THRESHOLD=0.8

# -----------------------------
# Neo4j Test Configuration
# -----------------------------
# Use different port for test database
TEST_NEO4J_URI=bolt://localhost:7688  # Note: Using 7688 for test
TEST_NEO4J_USER=neo4j
# Use your existing NEO4J_PASSWORD

# -----------------------------
# LLM Test Configuration
# -----------------------------
# Use your existing ANTHROPIC_API_KEY
TEST_CLAUDE_MODEL=claude-2

# Use your existing COHERE_API_KEY
TEST_COHERE_MODEL=command

# -----------------------------
# Vector Store Test Configuration
# -----------------------------
TEST_CHROMA_PERSIST_DIR=./.habitat/test_chroma
TEST_CHROMA_SERVER_LOG_LEVEL=DEBUG
```

## Setup Instructions

1. **Create Test Environment File**
   ```bash
   # Create a new .env.test file
   touch .env.test
   
   # Copy your external variables
   cp .env.example.bak .env.test
   
   # Append test-specific variables
   cat >> .env.test << 'EOL'
   # Test-specific variables here...
   EOL
   ```

2. **Directory Structure**
   ```bash
   # Create test directories
   mkdir -p .habitat/test_data
   mkdir -p .habitat/test_cache
   mkdir -p .habitat/embeddings
   mkdir -p .habitat/test_chroma
   ```

3. **Database Setup**
   ```bash
   # Start test Neo4j instance
   docker run \
     -p 7688:7687 \
     -e NEO4J_AUTH=neo4j/password \
     -e NEO4J_dbms_memory_pagecache_size=1G \
     -v $PWD/.habitat/test_neo4j:/data \
     neo4j:4.4
   ```

## Usage

1. **Activate Test Environment**
   ```bash
   # Source test environment
   export $(cat .env.test | grep -v '^#' | xargs)
   ```

2. **Run Tests**
   ```bash
   # Run with test configuration
   TEST_ENV=test pytest -v
   ```

## Important Notes

1. **External Services**
   - Using your existing API keys
   - Test-specific endpoints where needed
   - Separate test databases

2. **Data Isolation**
   - Test data in `.habitat/test_*`
   - Separate Neo4j port (7688)
   - Isolated ChromaDB instance

3. **Security**
   - No changes to production credentials
   - Test-specific configurations
   - Separate test endpoints

## Verification

```bash
# Verify environment
python -c "import os; print('TEST_ENV:', os.getenv('TEST_ENVIRONMENT'))"
python -c "import os; print('NEO4J_URI:', os.getenv('TEST_NEO4J_URI'))"
```

Remember:
- Keep test data isolated
- Use test-specific ports
- Maintain separate configurations
