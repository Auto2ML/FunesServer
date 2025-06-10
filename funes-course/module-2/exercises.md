# Module 2 Hands-On Exercises

## Exercise 1: Database Setup and Configuration (20 minutes)

### Objective
Set up PostgreSQL with pgvector extension and configure the Funes database environment.

### Step 1: Install PostgreSQL and pgvector

**Option A: Docker Setup (Recommended)**
```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: funes_postgres
    environment:
      POSTGRES_DB: funes_db
      POSTGRES_USER: funes_user
      POSTGRES_PASSWORD: funes_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF

# Start PostgreSQL
docker-compose up -d

# Verify container is running
docker ps
```

**Option B: Native Installation**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-server-dev-all

# Install pgvector
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Create user and database
sudo -u postgres createuser --interactive --pwprompt funes_user
sudo -u postgres createdb funes_db --owner=funes_user
```

### Step 2: Test Database Connection

```bash
# Test connection
psql -U funes_user -d funes_db -h localhost -c "SELECT version();"

# Enable pgvector extension
psql -U funes_user -d funes_db -h localhost -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify extension
psql -U funes_user -d funes_db -h localhost -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### ✅ Checkpoint
- PostgreSQL is running and accessible
- pgvector extension is installed and enabled
- Database connection established successfully

---

## Exercise 2: Schema Creation and Testing (25 minutes)

### Objective
Create the complete database schema with proper tables, indexes, and constraints.

### Step 1: Execute Schema Creation

```bash
# Run the schema creation script
psql -U funes_user -d funes_db -h localhost -f scripts/create_schema.sql

# Verify tables created
psql -U funes_user -d funes_db -h localhost -c "\dt"

# Check vector indexes
psql -U funes_user -d funes_db -h localhost -c "\di"
```

### Step 2: Test Vector Operations

```sql
-- Connect to database
psql -U funes_user -d funes_db -h localhost

-- Test basic vector operations
SELECT '[1,0,0]'::vector <-> '[1,0,0]'::vector as identical_distance;
SELECT '[1,0,0]'::vector <-> '[0,1,0]'::vector as orthogonal_distance;
SELECT '[1,0,0]'::vector <-> '[-1,0,0]'::vector as opposite_distance;

-- Test similarity search on sample data
SELECT content, embedding <-> '[0.1,0.2,0.3]'::vector as distance
FROM memories 
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector 
LIMIT 3;

-- Check index usage
EXPLAIN ANALYZE 
SELECT content FROM memories 
ORDER BY embedding <-> '[0.1,0.2,0.3]'::vector 
LIMIT 5;
```

### ✅ Checkpoint
- All tables created with proper constraints
- Vector indexes are functional
- Sample data queries work correctly
- Index usage confirmed in query plans

---

## Exercise 3: DatabaseManager Implementation (30 minutes)

### Objective
Build the core DatabaseManager class with connection pooling and CRUD operations.

### Step 1: Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install psycopg2-binary numpy
```

### Step 2: Implement Database Configuration

Create the configuration module:

```bash
mkdir -p config
```

Then implement the database configuration (code provided in the main module files).

### Step 3: Build DatabaseManager Class

Implement the complete DatabaseManager class with:
- Connection pooling
- Context managers
- CRUD operations
- Vector similarity search
- Error handling

### Step 4: Test Basic Operations

```python
# Test script
from database import DatabaseManager
from config.database import DB_CONFIG
import numpy as np

# Initialize database manager
db = DatabaseManager(DB_CONFIG)

# Test health check
print(f"Health check: {db.health_check()}")

# Test memory insertion
test_embedding = np.random.randn(384)
memory_id = db.insert_memory(
    content="Test memory insertion",
    embedding=test_embedding,
    source="test"
)
print(f"Inserted memory ID: {memory_id}")

# Test similarity search
results = db.retrieve_memories(test_embedding, top_k=3)
print(f"Found {len(results)} similar memories")
for result in results:
    print(f"- {result['content'][:50]}... (similarity: {result['similarity_score']:.3f})")

db.close()
```

### ✅ Checkpoint
- DatabaseManager class implemented with all methods
- Connection pooling working correctly
- CRUD operations functional
- Vector similarity search operational

---

## Exercise 4: Advanced Queries and Performance Testing (25 minutes)

### Objective
Implement advanced database operations and performance testing capabilities.

### Step 1: Test Data Generation

Run the test data generator to create sample data:

```python
from test_data_generator import TestDataGenerator
from database import DatabaseManager
from config.database import DB_CONFIG

db_manager = DatabaseManager(DB_CONFIG)
generator = TestDataGenerator(db_manager)

# Create test memories
memory_ids = generator.create_test_memories(100)
print(f"Created {len(memory_ids)} test memories")

# Test similarity search
generator.test_similarity_search()

db_manager.close()
```

### Step 2: Performance Benchmarking

Run performance tests to validate system capability:

```python
from performance_tests import PerformanceTester

db_manager = DatabaseManager(DB_CONFIG)
tester = PerformanceTester(db_manager)

# Test insert performance
insert_results = tester.test_insert_performance(100)

# Test search scaling
scaling_results = tester.test_search_performance_scaling()

# Test concurrent access
concurrent_results = tester.test_concurrent_access(3)

print("\n=== Performance Summary ===")
print(f"Average insert time: {insert_results['single_insert_avg']:.4f}s")
print(f"Search time at scale: {scaling_results['search_times'][-1]:.4f}s")
print(f"Concurrent query time: {concurrent_results['avg_query_time']:.4f}s")

db_manager.close()
```

### Step 3: Advanced Query Testing

Test advanced database operations:

```python
# Test filtered similarity search
results = db.retrieve_memories(
    query_embedding=np.random.randn(384),
    top_k=5,
    sources=['chat', 'tool'],
    min_importance=0.5,
    days_back=7
)

# Test tool embedding operations
tool_embedding = np.random.randn(384)
tool_id = db.store_tool_embedding(
    tool_name="advanced_test_tool",
    tool_description="Advanced testing tool for validation",
    tool_parameters={"param1": "string", "param2": "number"},
    embedding=tool_embedding
)

# Find relevant tools
relevant_tools = db.find_relevant_tools(tool_embedding, top_k=3)
for tool in relevant_tools:
    print(f"Tool: {tool['tool_name']}, Similarity: {tool['similarity_score']:.3f}")
```

### ✅ Checkpoint
- Test data generation working
- Performance benchmarks completed
- Advanced queries functional
- System performance meets expectations

---

## Exercise 5: Error Handling and Edge Cases (15 minutes)

### Objective
Test error handling, edge cases, and system robustness.

### Step 1: Error Condition Testing

```python
def test_error_handling():
    db = DatabaseManager(DB_CONFIG)
    
    # Test wrong embedding dimension
    try:
        wrong_embedding = np.random.randn(100)  # Wrong dimension
        db.insert_memory("Test", wrong_embedding)
        print("❌ Should have failed with wrong dimension")
    except Exception as e:
        print(f"✅ Correctly handled wrong dimension: {type(e).__name__}")
    
    # Test connection health during stress
    for i in range(50):
        if not db.health_check():
            print(f"❌ Health check failed at iteration {i}")
            break
    else:
        print("✅ Health check stable under stress")
    
    # Test memory statistics
    stats = db.get_memory_stats()
    print(f"✅ Memory stats: {stats}")
    
    db.close()

test_error_handling()
```

### Step 2: Edge Case Validation

```python
def test_edge_cases():
    db = DatabaseManager(DB_CONFIG)
    
    # Test empty content
    try:
        db.insert_memory("", np.random.randn(384))
        print("❌ Should reject empty content")
    except Exception:
        print("✅ Correctly rejected empty content")
    
    # Test null embedding search
    try:
        results = db.retrieve_memories(np.zeros(384), top_k=0)
        print(f"✅ Handled zero top_k: {len(results)} results")
    except Exception as e:
        print(f"✅ Correctly handled zero top_k: {e}")
    
    # Test very large embedding values
    large_embedding = np.ones(384) * 1000
    try:
        memory_id = db.insert_memory("Large embedding test", large_embedding)
        print(f"✅ Handled large embedding values: {memory_id}")
    except Exception as e:
        print(f"⚠️ Large embedding issue: {e}")
    
    db.close()

test_edge_cases()
```

### ✅ Checkpoint
- Error handling working correctly
- Edge cases handled gracefully
- System remains stable under stress
- Database integrity maintained

---

## Exercise 6: Complete System Validation (10 minutes)

### Objective
Run the complete test suite and validate all functionality.

### Step 1: Run Complete Test Suite

```bash
# Run the comprehensive test suite
python test_database.py
```

Expected output should show:
- ✅ All basic operations working
- ✅ Vector operations functional
- ✅ Performance within acceptable ranges
- ✅ Error handling robust
- ✅ Memory statistics accurate

### Step 2: Validate Database State

```sql
-- Check final database state
psql -U funes_user -d funes_db -h localhost << 'EOF'

-- Count records in each table
SELECT 'memories' as table_name, COUNT(*) as record_count FROM memories
UNION ALL
SELECT 'tool_embeddings', COUNT(*) FROM tool_embeddings
UNION ALL
SELECT 'documents', COUNT(*) FROM documents
UNION ALL
SELECT 'document_chunks', COUNT(*) FROM document_chunks;

-- Check index sizes
SELECT schemaname, tablename, indexname, idx_size
FROM (
    SELECT schemaname, tablename, indexname, 
           pg_size_pretty(pg_relation_size(indexrelid)) as idx_size
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
) t
ORDER BY tablename, indexname;

-- Verify vector dimensions
SELECT 
    'memories' as table_name,
    COUNT(*) as total_records,
    COUNT(embedding) as records_with_embeddings,
    COUNT(CASE WHEN vector_dims(embedding) = 384 THEN 1 END) as correct_dimensions
FROM memories
WHERE embedding IS NOT NULL;

EOF
```

### ✅ Final Checkpoint
- All tests passing
- Database in consistent state
- Performance metrics documented
- System ready for next module

---

## Module 2 Completion Summary

### What You've Built

1. **Production-Ready Database Layer**
   - PostgreSQL with pgvector extension
   - Proper schema design with constraints
   - Optimized indexing for vector operations

2. **Comprehensive DatabaseManager**
   - Connection pooling for performance
   - CRUD operations with error handling
   - Vector similarity search capabilities
   - Tool embedding management

3. **Performance Optimization**
   - Efficient query patterns
   - Index optimization strategies
   - Concurrent access handling
   - Performance monitoring tools

4. **Robust Testing Framework**
   - Unit tests for all operations
   - Performance benchmarking
   - Error condition testing
   - Edge case validation

### Key Skills Acquired

- ✅ Vector database design and implementation
- ✅ PostgreSQL advanced features (pgvector, JSONB, indexing)
- ✅ Python database programming with connection pooling
- ✅ Performance optimization for high-dimensional data
- ✅ Error handling and system robustness
- ✅ Test-driven development for database systems

### Performance Achievements

You should have achieved:
- Sub-10ms vector similarity searches
- 100+ memories/second insertion rate
- Stable performance with 1000+ memories
- Concurrent access without connection issues

### Ready for Module 3

Your database layer is now ready to support:
- Embedding generation and storage
- Memory lifecycle management
- Context building and retrieval
- Tool integration and selection

The solid foundation you've built will enable all subsequent modules of the Funes system.
