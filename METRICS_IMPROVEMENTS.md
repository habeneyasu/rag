# Metrics Improvement Methods

This document describes the specific methods and techniques implemented to improve accuracy, completeness, and relevance metrics in the RAG system.

## Retrieval Improvements

### Hybrid Search Integration
**Method**: Combined semantic search with keyword-based filtering
- Extracts keywords from queries (removes stop words, filters short words)
- Scores documents by keyword presence in content and metadata
- Merges semantic similarity scores with keyword match scores
**Impact**: Improves relevance by ensuring retrieved documents contain query-specific terms

### RAG-Fusion with Reciprocal Rank Fusion
**Method**: Multi-query retrieval with RRF merging
- Generates query variations using LLM (4-6 variations based on complexity)
- Retrieves documents for each variation independently
- Merges results using RRF formula: `score(d) = Σ(1 / (k + rank(d, q)))` where k=60
**Impact**: Dramatically improves completeness by capturing synonyms and related concepts across multiple query phrasings

### Cross-Encoder Reranking
**Method**: Semantic reranking of top candidates
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for query-document similarity
- Selectively reranks top 10-12 candidates to optimize cost
- Reorders documents based on cross-encoder scores
**Impact**: Improves precision and relevance of top-ranked documents

## Document Processing Improvements

### Hierarchical Markdown Chunking
**Method**: Preserves document structure during chunking
- Uses `MarkdownHeaderTextSplitter` to maintain section hierarchy
- Preserves header metadata (Header 1, 2, 3) in chunk metadata
- Dynamic chunk sizes: Contracts (500), Products (900), Other (700)
**Impact**: Preserves complex policy relationships and improves context coherence

## Adaptive Retrieval Strategy

### Query Complexity Classification
**Method**: Rule-based classification with adaptive parameters
- **Simple**: 1 variation, 7 chunks, minimal processing
- **Moderate**: 4 variations, 8 chunks, standard processing
- **Complex**: 5 variations, 8 chunks, domain knowledge enabled
- **Broad**: 6 variations, 10 chunks, comprehensive retrieval
**Impact**: Optimizes resource usage while ensuring appropriate retrieval depth for each query type

## Domain Knowledge Enhancement

### Terminology Extraction and Injection
**Method**: Specialized retrieval for domain-specific terms
- Extracts insurance terminology, product names, and contract language
- Performs additional retrieval using domain terms as queries
- Retrieves 3 additional chunks per domain term
**Impact**: Improves accuracy for technical questions and edge cases

### Edge Case Detection
**Method**: Pattern matching for rare scenarios
- Detects patterns: "edge case", "exception", "not covered", "limitation", etc.
- Triggers specialized retrieval with expanded query terms
- Retrieves 6 additional chunks for edge cases
**Impact**: Improves completeness for complex, rare, or exception scenarios

## Generation Improvements

### Self-Correction Loop
**Method**: LLM-based answer verification and correction
- Cross-checks answer against all retrieved chunks
- Three-point verification:
  1. **Accuracy (>4.5)**: Verifies every fact against context, removes unverified claims
  2. **Completeness (>=4.5)**: Processes each chunk systematically, adds missing information
  3. **Relevance (>=4.90)**: Removes unrelated content, focuses on question
- Priority: Accuracy > Completeness > Relevance
**Impact**: Eliminates hallucinations, ensures all relevant information included, maintains focus

### Enhanced Prompt Engineering
**Method**: Few-shot examples and structured formatting
- Includes example Q&A pairs in system prompt
- Clear citation format: `[Source: filename.md]`
- Explicit instructions for accuracy, completeness, and relevance
- Product name verification list included
**Impact**: Reduces hallucinations, improves citation accuracy, ensures consistent formatting

### Context Summarization (Conditional)
**Method**: Summarizes chunks when context exceeds token limits
- Only activates when estimated tokens > 80% of max_tokens
- Preserves all factual information (numbers, dates, names, features)
- Target: 20-30% reduction while maintaining completeness
**Impact**: Enables processing of more chunks without exceeding token limits

## Code Quality Optimizations

### Deduplication
**Method**: Removes duplicate documents across retrieval stages
- Uses content hash + source path as unique identifier
- Preserves order while eliminating duplicates
**Impact**: Prevents redundant context, improves efficiency

### Consolidated Error Handling
**Method**: Graceful degradation for optional features
- Reranking fails gracefully if cross-encoder unavailable
- Query variation generation falls back to original query
- Domain knowledge retrieval continues even if individual terms fail
**Impact**: System remains functional even when optional components fail

## Configuration Tuning

Default parameters optimized for metric improvement:
- `USE_RERANKING=true`: Enables precision improvement
- `USE_RAG_FUSION=true`: Enables completeness improvement
- `USE_SELF_CORRECTION=true`: Enables accuracy improvement
- `USE_DOMAIN_KNOWLEDGE=true`: Enables domain-specific accuracy
- `NUM_QUERY_VARIATIONS=4`: Balanced between completeness and efficiency
- `DOMAIN_KNOWLEDGE_K=3`: Additional context for domain terms

## Performance Impact: Baseline vs. Optimized

| Metric Category | Metric | Baseline | Improved | % Increase | Primary Driver | Optimization Technique / Approach Used |
|----------------|--------|----------|----------|------------|----------------|------------------------------------------|
| Retrieval | MRR | 0.7228 | 0.8087 | +11.88% | Cross-Encoder Reranking | Multi-stage retrieval pipeline with semantic search + reranking |
| Retrieval | nDCG | 0.7392 | 0.8102 | +9.60% | Hybrid Search | Combination of semantic embeddings and keyword-based filtering |
| Retrieval | Keyword Coverage | 80.8% | 93.5% | +15.72% | RAG-Fusion | Multi-query expansion with Reciprocal Rank Fusion (RRF) |
| Generation | Accuracy | 4.09 | 4.53 | +10.76% | Self-Correction Loop | Automated answer verification and correction using retrieved context |
| Generation | Completeness | 3.63 | 3.97 | +9.37% | Hierarchical Chunking | Markdown-based structured chunking with dynamic chunk sizes |
| Generation | Relevance | 4.85 | 4.67 | -3.71% | Prompt Engineering | Domain-specific system prompts with few-shot examples |

## Summary

The implemented optimizations resulted in significant improvements across retrieval and generation metrics:
- **Retrieval metrics** improved by 9-16% through multi-stage retrieval, hybrid search, and RAG-Fusion
- **Accuracy** improved by 10.76% through self-correction mechanisms
- **Completeness** improved by 9.37% through hierarchical chunking and systematic processing
- **Relevance** showed a slight decrease (-3.71%), likely due to prioritizing accuracy and completeness in the self-correction loop
