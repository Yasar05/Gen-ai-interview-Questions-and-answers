# RAG and Retrieval - Generative AI Interview Questions

**Powered by [Euron](https://euron.one/) - Your AI Innovation Partner**

## Questions 66-80: RAG and Retrieval Systems

---

### 66. Design a RAG system for 1 million documents.

**Answer:**

**Architecture:**
Documents → Chunking → Embeddings → Vector DB → Retrieval → Generation → Response

**Implementation Components:**
- **Document Processing**: Chunk documents into manageable pieces
- **Embedding Generation**: Create vector representations using embedding models
- **Vector Storage**: Store embeddings in vector databases (Pinecone, Weaviate)
- **Query Processing**: Generate query embeddings and retrieve relevant chunks
- **Response Generation**: Use retrieved context to generate responses

**Scalability Considerations:**
- **Distributed Processing**: Use multiple workers
- **Batch Processing**: Process documents in batches
- **Incremental Updates**: Add new documents efficiently
- **Caching**: Cache frequent queries
- **Load Balancing**: Distribute query load

---

### 67. What are the different chunking strategies for RAG?

**Answer:**

**Chunking Strategies:**

**1. Fixed-size Chunking:**
- **Process**: Split text into fixed-size chunks with overlap
- **Parameters**: Chunk size (e.g., 512 tokens), overlap (e.g., 50 tokens)
- **Benefits**: Simple and consistent chunk sizes
- **Use Case**: General-purpose document processing

**2. Semantic Chunking:**
- **Process**: Group sentences based on semantic similarity. Instead of slicing by size, we slice based on meaning. The system groups sentences that are semantically related by analyzing:topic similarity, sentence embeddings, paragraph boundaries. Semantic chunking ensures each chunk stays within one topic.
- **Parameters**: Similarity threshold (e.g., 0.8)
- **Benefits**: Maintains semantic coherence within chunks
- **Use Case**: Documents with clear semantic boundaries

**3. Hierarchical Chunking:**
- **Process**: Create multiple levels of chunks (sections, paragraphs, sentences).
- Level 1 → sections
Level 2 → subsections
Level 3 → paragraphs
Level 4 → sentences
- **Structure**: Hierarchical organization of document content
- **Benefits**: Flexible retrieval at different granularity levels
- **Use Case**: Complex documents with clear hierarchical structure

**4. Sliding Window Chunking:**
- **Process**: Create overlapping chunks with sliding window approach. This method uses overlapping windows. This ensures the context flows smoothly across chunks.
- **Parameters**: Window size (e.g., 512 tokens), step size (e.g., 256 tokens)
- **Benefits**: Ensures context continuity across chunks
- **Use Case**: Sequential documents where context matters

**Best Practices:**
- **Overlap**: Include context between chunks
- **Size**: Balance context and specificity
- **Semantics**: Respect semantic boundaries
- **Metadata**: Include source information
- **Testing**: Evaluate chunk quality

---

### 68. How do you implement hybrid search in RAG systems?

**Answer:**

Hybrid search combines dense (semantic) and sparse (keyword) retrieval for better results.Hybrid search means using two different retrieval methods together:
Dense retrieval → finds text based on meaning (semantic similarity)
Sparse retrieval → finds text based on exact words (keyword matching)
Why? Because some queries need exact matching (“invoice 2023-44”), while others need semantic matching (“symptoms of high sugar levels”).
Hybrid search mixes these scores using weights like:
70% semantic + 30% keyword or 50/50 or tuned based on use case
**Implementation Components:**
- **Dense Retriever**: Semantic search using embeddings
- **Sparse Retriever**: Keyword-based search (TF-IDF, BM25)
- **Score Combination**: Weighted combination of dense and sparse scores
- **Result Ranking**: Rank results by combined scores
- **Normalization**: Semantic scores and keyword scores come from different mathematical scales. Normalize scores from different retrievers

**Benefits:**
- **Comprehensive**: Covers both semantic and keyword matches
- **Robust**: Handles various query types
- **Flexible**: Adjustable weights
- **Quality**: Better overall results

---

### 69. What is the difference between dense and sparse retrieval?

**Answer:**

**Dense Retrieval:**
- **Method**: Uses dense vector representations
- **Model**: Neural networks (BERT, RoBERTa)
- **Embeddings**: High-dimensional vectors
- **Similarity**: Cosine similarity
- **Strengths**: Semantic understanding, context

**Sparse Retrieval:**
- **Method**: Uses sparse vector representations
- **Model**: TF-IDF, BM25
- **Embeddings**: Sparse vectors
- **Similarity**: Dot product
- **Strengths**: Exact matches, keywords

**Comparison:**
| Aspect | Dense | Sparse |
|--------|-------|--------|
| **Semantics** | High | Low |
| **Keywords** | Low | High |
| **Speed** | Medium | Fast |
| **Memory** | High | Low |
| **Quality** | High | Medium |

**When to Use:**
- **Dense**: Semantic search, context understanding
- **Sparse**: Keyword search, exact matches
- **Hybrid**: Best of both worlds

---

### 70. How do you implement reranking in RAG systems?

**Answer:**

Reranking improves retrieval quality by reordering results using more sophisticated models.

**Implementation Components:**
- **Reranking Model**: Sophisticated model for scoring
- **Document Scoring**: Score each document against query
- **Score Ranking**: Sort documents by relevance scores
- **Top-K Selection**: Select top-k most relevant documents
- **Cross-encoder**: Use cross-encoder for better accuracy

**Reranking Models:**
- **Cross-encoders**: BERT-based rerankers
- **Bi-encoders**: Separate query/document encoders
- **Learning-to-rank**: ML models for ranking
- **Neural ranking**: Deep learning approaches

---

### 71. What are the best practices for vector database selection?

**Answer:**

**Vector Database Options:**

**1. Pinecone:**
- **Pros**: Managed service, easy setup
- **Cons**: Cost, vendor lock-in
- **Use Case**: Production applications

**2. Weaviate:**
- **Pros**: Open-source, flexible
- **Cons**: Setup complexity
- **Use Case**: Custom deployments

**3. Chroma:**
- **Pros**: Simple, lightweight
- **Cons**: Limited features
- **Use Case**: Development, small projects

**4. Qdrant:**
- **Pros**: Performance, features
- **Cons**: Learning curve
- **Use Case**: High-performance applications

**Selection Criteria:**
- **Scale**: Number of vectors
- **Performance**: Query speed
- **Features**: Filtering, metadata
- **Cost**: Pricing model
- **Maintenance**: Self-hosted vs managed

**Best Practices:**
- **Start Simple**: Begin with basic solutions
- **Scale Gradually**: Upgrade as needed
- **Monitor Performance**: Track query times
- **Plan for Growth**: Consider future needs
- **Test Thoroughly**: Evaluate options

---

### 72. How do you handle metadata filtering in RAG?

**Answer:**

Metadata filtering allows querying by document properties alongside semantic search.Metadata filtering means restricting retrieval results based on information attached to each chunk — like document type, date, author, language, department, or category.


**Implementation Components:**
- **Filter Query Builder**: Build filter queries from criteria. Translate requirements into filters. User asks: “Show me finance policies updated after 2022.”
Filter Query Builder creates a filter with 2022 year.
- **Vector Database Integration**: Apply filters inside the vector search.Vector DBs like Qdrant, Weaviate, Pinecone, Milvus allow filtered vector search.This means: First apply metadata filters. Then run semantic search inside the filtered subset. Why? It avoids retrieving irrelevant chunks entirely.
Example: Search only inside: year = 2023,department = Finance
Instead of searching all documents.
- **Metadata Matching**: Match documents against filter criteria. Check if each chunk satisfies the filter. The system checks: Does this chunk’s metadata match the filter? If yes → include it in the candidate pool, If not → ignore it
- **Result Filtering**: Filter results based on metadata. Even after retrieval, you may want to run post-filtering:Remove chunks missing required metadata. Remove private, restricted, or unsafe categories. Down-rank or discard irrelevant document types
- **Query Optimization**: Optimize filtered queries for performance. Make filtered searches fast & efficient. eg: Search only the "Finance" partition instead of scanning all chunks.

**Filter Types:**
- **Exact Match**: doc_type = "pdf"
- **Range**: date >= "2023-01-01"
- **Inclusion**: author in ["John", "Jane"]
- **Exclusion**: category != "private"
- **Combination**: Multiple conditions

---

### 73. What is the purpose of query expansion in RAG?

**Answer:**

Query expansion improves retrieval by generating additional query terms and variations.

**Techniques:**
1. **Synonym Expansion**: Add synonyms
2. **Related Terms**: Add related concepts
3. **Query Reformulation**: Rewrite queries
4. **Multi-lingual**: Add translations
5. **Contextual**: Add context terms

**Implementation:**
```python
class QueryExpansion:
    def __init__(self, expansion_model):
        self.expansion_model = expansion_model
    
    def expand_query(self, query):
        # Generate synonyms
        synonyms = self.get_synonyms(query)
        
        # Generate related terms
        related_terms = self.get_related_terms(query)
        
        # Combine expansions
        expanded_query = f"{query} {' '.join(synonyms)} {' '.join(related_terms)}"
        
        return expanded_query
    
    def get_synonyms(self, query):
        # Use WordNet or similar
        synonyms = []
        for word in query.split():
            syns = wordnet.synsets(word)
            synonyms.extend([syn.lemmas()[0].name() for syn in syns])
        return synonyms
```

**Benefits:**
- **Recall**: Find more relevant documents
- **Robustness**: Handle query variations
- **Coverage**: Cover different aspects
- **Quality**: Better retrieval results

---

### 74. How do you implement multi-hop reasoning in RAG?

**Answer:**

Multi-hop reasoning chains multiple retrieval steps to answer complex questions.

**Implementation Components:**
- **Multi-hop Retriever**: Chain multiple retrieval steps
- **Context Accumulation**: Build context across hops
- **Answer Validation**: Check if enough information is available
- **Query Generation**: Generate follow-up queries
- **Reasoning Chain**: Manage the reasoning process

**Benefits:**
- **Complex Questions**: Handle multi-step reasoning
- **Better Answers**: More comprehensive responses
- **Context Building**: Progressive information gathering
- **Accuracy**: Higher quality results

---

### 75. What are the challenges of RAG in production?

**Answer:**

**Technical Challenges:**
- **Latency**: Real-time response requirements
- **Scalability**: Handle high query volumes
- **Consistency**: Maintain data freshness
- **Quality**: Ensure retrieval accuracy
- **Cost**: Manage computational expenses

**Data Challenges:**
- **Freshness**: Keep data up-to-date
- **Quality**: Ensure data accuracy
- **Coverage**: Complete information coverage
- **Bias**: Avoid biased results
- **Privacy**: Handle sensitive data

**Operational Challenges:**
- **Monitoring**: Track system performance
- **Debugging**: Identify and fix issues
- **Updates**: Deploy new versions
- **Maintenance**: Keep systems running
- **Support**: Handle user issues

**Solutions:**
- **Caching**: Reduce latency
- **Load Balancing**: Distribute load
- **Monitoring**: Track metrics
- **Testing**: Comprehensive testing
- **Documentation**: Clear documentation

---

### 76. How do you evaluate RAG system performance?

**Answer:**

**Evaluation Metrics:**

**1. Retrieval Metrics:**
- **Precision**: Relevant documents retrieved
- **Recall**: Relevant documents found
- **F1 Score**: Harmonic mean of precision and recall
- **NDCG**: Normalized Discounted Cumulative Gain

**2. Generation Metrics:**
- **BLEU**: N-gram overlap with reference
- **ROUGE**: Recall-oriented evaluation
- **BERTScore**: Semantic similarity
- **Human Evaluation**: Manual assessment

**Implementation Components:**
- **Retrieval Evaluation**: Measure retrieval quality
- **Generation Evaluation**: Measure generation quality
- **Metric Computation**: Compute evaluation metrics
- **Score Aggregation**: Aggregate scores across queries
- **Performance Analysis**: Analyze system performance

**Best Practices:**
- **Multiple Metrics**: Use various evaluation methods
- **Human Evaluation**: Include human assessment
- **A/B Testing**: Compare different approaches
- **Continuous Monitoring**: Track performance over time
- **Benchmarking**: Compare with baselines

---

### 77. What is the difference between RAG and fine-tuning?

**Answer:**

**RAG (Retrieval-Augmented Generation):**
- **Method**: Retrieve relevant information, then generate
- **Data**: External knowledge base
- **Training**: Minimal training required
- **Flexibility**: Easy to update knowledge
- **Cost**: Lower computational cost
- **Use Case**: Knowledge-intensive tasks

**Fine-tuning:**
- **Method**: Train model on specific data
- **Data**: Task-specific training data
- **Training**: Extensive training required
- **Flexibility**: Harder to update
- **Cost**: Higher computational cost
- **Use Case**: Domain-specific tasks

**Comparison:**
| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge** | External | Internal |
| **Training** | Minimal | Extensive |
| **Updates** | Easy | Hard |
| **Cost** | Low | High |
| **Performance** | Good | Better |

**When to Use:**
- **RAG**: Knowledge-intensive, frequently updated
- **Fine-tuning**: Domain-specific, stable requirements

---

### 78. How do you implement iterative retrieval?

**Answer:**

Iterative retrieval refines search queries based on initial results to improve retrieval quality.

**Implementation:**
```python
class IterativeRetrieval:
    def __init__(self, retriever, query_expander, max_iterations=3):
        self.retriever = retriever
        self.query_expander = query_expander
        self.max_iterations = max_iterations
    
    def retrieve(self, query):
        current_query = query
        all_results = []
        
        for iteration in range(self.max_iterations):
            # Retrieve documents
            results = self.retriever.search(current_query, top_k=5)
            all_results.extend(results)
            
            # Check if we have enough information
            if self.has_sufficient_info(query, all_results):
                break
            
            # Expand query based on results
            current_query = self.expand_query(query, results)
        
        # Deduplicate and rank results
        final_results = self.deduplicate_and_rank(all_results)
        return final_results
    
    def expand_query(self, original_query, results):
        # Extract key terms from results
        key_terms = self.extract_key_terms(results)
        
        # Expand query
        expanded_query = f"{original_query} {' '.join(key_terms)}"
        return expanded_query
    
    def has_sufficient_info(self, query, results):
        # Check if results contain enough information
        # This could be a learned model or heuristic
        return len(results) >= 10
```

**Benefits:**
- **Better Coverage**: Find more relevant documents
- **Query Refinement**: Improve search quality
- **Adaptive**: Adjust to query complexity
- **Quality**: Higher retrieval accuracy

---

### 79. What are the best practices for RAG prompt engineering?

**Answer:**

**Prompt Engineering Best Practices:**

**1. Clear Instructions:**
- Provide clear role definition
- Specify context usage requirements
- Define fallback behavior
- Include answer format requirements

**2. Context Formatting:**
- Structure context documents clearly
- Number documents for reference
- Separate different sources
- Maintain readability

**3. Answer Formatting:**
- Specify answer format
- Include source references
- Define citation style
- Ensure consistency

**4. Error Handling:**
- Define error responses
- Specify when to admit uncertainty
- Provide fallback messages
- Handle edge cases

**5. Quality Control:**
- Specify accuracy requirements
- Define source citation standards
- Set conciseness guidelines
- Ensure comprehensive coverage

**Best Practices:**
- **Be Specific**: Clear, detailed instructions
- **Format Context**: Structure information clearly
- **Handle Errors**: Graceful failure handling
- **Test Prompts**: Iterate and improve
- **Monitor Quality**: Track response quality

---

### 80. How do you handle RAG system scalability?

**Answer:**

**Scalability Strategies:**

**1. Horizontal Scaling:**
- **Worker Distribution**: Distribute queries across multiple workers
- **Load Balancing**: Balance load across available workers
- **Worker Management**: Manage worker lifecycle and health
- **Query Routing**: Route queries to appropriate workers

**2. Caching:**
- **Response Caching**: Cache frequent query responses
- **Cache Management**: Implement LRU or similar cache policies
- **Cache Invalidation**: Handle cache updates and invalidation
- **Performance Optimization**: Optimize cache hit rates

**3. Asynchronous Processing:**
- **Async Query Processing**: Process multiple queries concurrently
- **Task Management**: Manage async tasks efficiently
- **Result Aggregation**: Aggregate results from async operations
- **Performance Scaling**: Scale with concurrent requests

**4. Database Optimization:**
- **Index Optimization**: Optimize vector database indexes
- **Query Optimization**: Optimize query performance
- **Connection Pooling**: Manage database connections
- **Resource Management**: Optimize resource usage

**Best Practices:**
- **Load Balancing**: Distribute queries
- **Caching**: Cache frequent queries
- **Async Processing**: Handle multiple queries
- **Database Optimization**: Optimize vector search
- **Monitoring**: Track performance metrics

---

## Summary

RAG systems are crucial for building knowledge-intensive GenAI applications. Focus on understanding retrieval techniques, generation quality, and system scalability.

**Key Takeaways:**
- Master retrieval techniques
- Understand generation quality
- Focus on system scalability
- Practice with real implementations
- Stay updated with new developments

**Next Steps:**
- Build RAG systems
- Experiment with different approaches
- Optimize for production
- Learn from real deployments
- Contribute to the community

---

**Powered by [Euron](https://euron.one/) - Empowering AI Innovation Worldwide**
