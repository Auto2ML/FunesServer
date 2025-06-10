# Module 2: Database Layer Implementation

This module implements the core database layer for the Funes memory system using PostgreSQL with pgvector extension.

## Files Overview

- `database.py` - Main DatabaseManager class with connection pooling and vector operations
- `config/database.py` - Database configuration and connection parameters
- `scripts/create_schema.sql` - Complete database schema with tables and indexes
- `test_data_generator.py` - Generate test data and validate similarity search
- `performance_tests.py` - Performance benchmarking and scaling tests
- `test_database.py` - Complete test suite for all functionality
- `requirements.txt` - Python dependencies
- `setup.sh` - Installation and setup script

## Key Features Implemented

### Database Management
- ✅ Connection pooling with psycopg2
- ✅ Context managers for safe connection handling
- ✅ Transaction management with rollback support
- ✅ Health monitoring and error handling

### Vector Operations
- ✅ High-dimensional vector storage (384 dimensions)
- ✅ Cosine similarity search with pgvector
- ✅ HNSW indexing for performance optimization
- ✅ Batch embedding operations

### Memory Storage
- ✅ Structured memory table with metadata
- ✅ Session and user-based organization
- ✅ Importance scoring for memory prioritization
- ✅ Source tracking (chat, tool, system, document)

### Tool Integration
- ✅ Tool embedding storage for vector-based selection
- ✅ Usage tracking and analytics
- ✅ Parameter schema storage in JSONB

### Performance Features
- ✅ Connection pooling for concurrent access
- ✅ Prepared statements for frequent queries
- ✅ Index optimization strategies
- ✅ Performance benchmarking tools

## Quick Start

1. **Setup Environment**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source venv/bin/activate
   ```

2. **Create Database Schema**
   ```bash
   psql -U funes_user -d funes_db -h localhost -f scripts/create_schema.sql
   ```

3. **Run Tests**
   ```bash
   python test_database.py
   ```

## Learning Objectives Achieved

By completing this module, you will have:

- ✅ Built a production-ready database layer with vector capabilities
- ✅ Implemented efficient similarity search with proper indexing
- ✅ Created robust connection management with pooling
- ✅ Developed comprehensive error handling and recovery
- ✅ Established performance testing and optimization strategies
- ✅ Mastered PostgreSQL pgvector extension usage

## Next Steps

After completing Module 2, proceed to Module 3 to build the embedding system and memory manager that will use this database layer.

## Performance Benchmarks

Expected performance on modern hardware:
- Vector similarity search: <10ms for databases up to 100K memories
- Memory insertion: <5ms per memory with embedding
- Concurrent access: 50+ queries/second with connection pooling
- Index build time: ~1 minute for 10K memories

## Troubleshooting

### Common Issues

1. **pgvector Extension Not Found**
   - Ensure pgvector is properly installed
   - Check PostgreSQL version compatibility
   - Verify extension is enabled: `CREATE EXTENSION vector;`

2. **Connection Pool Exhaustion**
   - Increase pool size in configuration
   - Check for connection leaks in application code
   - Monitor connection usage patterns

3. **Slow Vector Queries**
   - Verify HNSW index exists and is being used
   - Check index parameters (m, ef_construction)
   - Consider rebuilding indexes with VACUUM ANALYZE

### Debug Commands

```sql
-- Check extension status
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Verify index usage
EXPLAIN ANALYZE SELECT * FROM memories ORDER BY embedding <-> '[...]' LIMIT 5;

-- Check table statistics
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables WHERE tablename IN ('memories', 'tool_embeddings');
```
