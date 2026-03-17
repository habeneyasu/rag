import os
import re
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

# Logging configuration (consolidated from logger.py)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# Console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for persistent logs
file_handler = logging.FileHandler(LOG_DIR / "rag.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False

# Multi-stage retrieval: Cross-encoder reranking
try:
    from cross_encoder import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("cross-encoder not installed. Install with: uv add cross-encoder")

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Configure OpenRouter from .env file
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
max_tokens = int(os.getenv("MAX_TOKENS", "2000"))  # Convert to int

INITIAL_RETRIEVAL_K = 12
FINAL_RETRIEVAL_K = 12
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"

NUM_QUERY_VARIATIONS = int(os.getenv("NUM_QUERY_VARIATIONS", "4"))
CHUNKS_PER_QUERY = int(os.getenv("CHUNKS_PER_QUERY", "8"))
USE_RAG_FUSION = os.getenv("USE_RAG_FUSION", "true").lower() == "true"
RRF_K = 60

USE_CONTEXT_SUMMARIZATION = os.getenv("USE_CONTEXT_SUMMARIZATION", "false").lower() == "true"
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "15"))

USE_SELF_CORRECTION = os.getenv("USE_SELF_CORRECTION", "true").lower() == "true"
USE_CRITIQUE_SCORING = os.getenv("USE_CRITIQUE_SCORING", "false").lower() == "true"
CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL", MODEL)

USE_DOMAIN_KNOWLEDGE = os.getenv("USE_DOMAIN_KNOWLEDGE", "true").lower() == "true"
DOMAIN_KNOWLEDGE_K = int(os.getenv("DOMAIN_KNOWLEDGE_K", "3"))

reranker = None
if RERANKER_AVAILABLE and USE_RERANKING:
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker initialized")
    except Exception as e:
        logger.warning(f"Failed to load reranker: {e}")
        reranker = None

SYSTEM_PROMPT = """You are Insurellm's Expert Assistant. Provide accurate, complete, and relevant information about Insurellm and its insurance products.

**Principles:**

1. **Accuracy**: Answer ONLY from the provided context. Verify every fact, number, date, and product name. If uncertain, state: "The documents do not contain this information." Product names: Carllm, Homellm, Lifellm, Healthllm, Bizllm, Markellm, Claimllm, Rellm.

2. **Completeness**: Process each context chunk systematically. Include all relevant details from every chunk. For "all products" questions, list all 8 products.

3. **Relevance**: Answer the specific question directly. Minimize unrelated content.

**Format**: Use [Source: filename.md] citations. Structure with sections/bullets. If information is missing, state: "The documents do not contain this information."

**Examples:**

Q: What is Carllm?
A: Carllm is an innovative auto insurance product developed by Insurellm, designed to streamline how insurance companies offer coverage to customers. It uses AI to deliver personalized auto insurance solutions. [Source: products/Carllm.md]

Q: When was Insurellm founded?
A: Insurellm was founded by Avery Lancaster in 2015 as an insurance tech startup. [Source: company/about.md]

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Hybrid retrieval: Combine semantic search + keyword filtering
def extract_keywords(question: str) -> List[str]:
    """Extract keywords from the question for filtering."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who", "when", "where", "how", "why", "which", "about", "for", "with", "from"}
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def keyword_filter(docs: List[Document], keywords: List[str]) -> List[Document]:
    """Filter documents by keyword presence (hybrid retrieval)."""
    if not keywords:
        return docs
    
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        exact_matches = sum(1 for keyword in keywords if keyword in content_lower)
        metadata_text = " ".join(str(v).lower() for v in doc.metadata.values())
        metadata_matches = sum(1 for keyword in keywords if keyword in metadata_text)
        score = exact_matches + (metadata_matches * 0.5) + 0.01
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


def generate_query_variations(original_query: str, num_variations: int = None) -> List[str]:
    """Generate query variations for RAG-Fusion."""
    if not USE_RAG_FUSION:
        return [original_query]
    
    # Use adaptive number if provided, otherwise use default
    target_variations = num_variations if num_variations is not None else NUM_QUERY_VARIATIONS
    
    # For simple queries (1 variation), skip LLM call and return original
    if target_variations == 1:
        return [original_query]
    
    # Product names and insurance terminology for context
    products = ["Carllm", "Homellm", "Lifellm", "Healthllm", "Bizllm", "Markellm", "Claimllm", "Rellm"]
    insurance_terms = ["insurance", "policy", "coverage", "premium", "claim", "underwriting", "risk", "pricing"]
    
    prompt = f"""Generate {target_variations} diverse query variations for the following question about Insurellm.
Each variation should:
1. Use different wording and phrasing
2. Include relevant synonyms and insurance terminology
3. Consider product-specific terms when relevant (Carllm, Homellm, Lifellm, Healthllm, Bizllm, Markellm, Claimllm, Rellm)
4. Paraphrase the question in different ways
5. Maintain the core intent of the original question

Original question: {original_query}

Return ONLY a numbered list of {target_variations} query variations, one per line, without any additional text:"""
    
    try:
        response = query_llm.invoke([HumanMessage(content=prompt)])
        variations_text = response.content.strip()
        
        # Parse variations from the response
        variations = []
        for line in variations_text.split('\n'):
            # Remove numbering (1., 2., etc.)
            line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if line and len(line) > 10:  # Filter out very short lines
                variations.append(line)
        
        # Always include original query as first variation
        if original_query not in variations:
            variations.insert(0, original_query)
        
        # Ensure we have at least target_variations
        while len(variations) < target_variations:
            variations.append(original_query)  # Fallback to original
        
        return variations[:target_variations]
    
    except Exception as e:
        logger.warning(f"Query variation generation failed: {e}")
        return [original_query]


def reciprocal_rank_fusion(all_results: List[List[Document]]) -> List[Document]:
    """
    Merge multiple retrieval results using Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(d) = Σ(1 / (k + rank(d, q)))
    where k is a constant (typically 60) and rank(d, q) is the rank of document d in query q's results.
    
    Result: Completeness ↑ dramatically; fewer missed facts across products.
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    
    # Calculate RRF scores for each document across all queries
    for query_results in all_results:
        for rank, doc in enumerate(query_results, start=1):
            # Use a unique identifier for each document (content hash or source+content)
            doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
            doc_map[doc_id] = doc
            # RRF formula: 1 / (k + rank)
            doc_scores[doc_id] += 1.0 / (RRF_K + rank)
    
    # Sort documents by RRF score (descending)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return documents in order of RRF score
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def rerank_documents(question: str, docs: List[Document], max_rerank: int = 10) -> List[Document]:
    """
    Rerank documents using cross-encoder for better relevance.
    Optimization 2: Selective Reranking - only rerank top candidates to reduce cost.
    Improved: Increased max_rerank to 12 for better quality filtering.
    
    Args:
        question: The query
        docs: Documents to rerank
        max_rerank: Maximum number of documents to rerank (increased to 12 for better quality)
    """
    if not reranker or not docs:
        return docs
    
    if len(docs) <= 3:
        return docs
    
    try:
        rerank_candidates = docs[:max_rerank]
        remaining_docs = docs[max_rerank:]
        pairs = [[question, doc.page_content] for doc in rerank_candidates]
        scores = reranker.predict(pairs)
        scored_docs = list(zip(scores, rerank_candidates))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs]
        return reranked + remaining_docs
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return docs


class QueryComplexity:
    """Query complexity classification for adaptive RAG."""
    SIMPLE = "simple"      # Fast-path: minimal processing
    MODERATE = "moderate"  # Standard processing
    COMPLEX = "complex"    # Deep-path: full processing
    BROAD = "broad"        # Comprehensive: all products/overview


def classify_query(question: str) -> QueryComplexity:
    """Classify query complexity using rule-based heuristics."""
    question_lower = question.lower()
    
    # Broad queries: asking about all products, company overview, comprehensive info
    broad_indicators = [
        "all products", "products", "product suite", "offerings", "tell me about",
        "overview", "company and", "insurellm and", "everything", "complete",
        "comprehensive", "list all", "what are all"
    ]
    if any(indicator in question_lower for indicator in broad_indicators):
        return QueryComplexity.BROAD
    
    # Complex queries: multi-part, technical terms, edge cases
    complex_indicators = [
        " and ", " or ", " both ", " also ", " as well as ",  # Multi-part
        "how does", "how do", "explain", "describe", "compare",  # Explanatory
        "contract", "pricing", "cost", "tier", "plan",  # Technical
        "edge case", "exception", "special", "unusual"  # Edge cases
    ]
    technical_terms = [
        "underwriting", "actuarial", "reinsurance", "indemnity", "subrogation",
        "endorsement", "rider", "exclusion", "beneficiary", "annuity"
    ]
    if (any(indicator in question_lower for indicator in complex_indicators) or
        any(term in question_lower for term in technical_terms)):
        return QueryComplexity.COMPLEX
    
    simple_indicators = [
        "what is", "who is", "when was", "where is", "how many",
        "what is the", "who founded", "when did"
    ]
    if (any(indicator in question_lower for indicator in simple_indicators) and
        len(question.split()) <= 5):
        return QueryComplexity.SIMPLE
    
    # Default to moderate
    return QueryComplexity.MODERATE


def get_adaptive_parameters(query_complexity: QueryComplexity) -> Dict[str, int]:
    """Get adaptive retrieval parameters based on query complexity."""
    if query_complexity == QueryComplexity.SIMPLE:
        return {
            "num_query_variations": 1,
            "chunks_per_query": 7,
            "initial_retrieval_k": 10,
            "final_retrieval_k": 8,
            "use_reranking": True,
            "use_domain_knowledge": False,
            "use_self_correction": True,
        }
    elif query_complexity == QueryComplexity.MODERATE:
        return {
            "num_query_variations": 4,
            "chunks_per_query": 8,
            "initial_retrieval_k": 12,
            "final_retrieval_k": 10,
            "use_reranking": True,
            "use_domain_knowledge": False,
            "use_self_correction": True,
        }
    elif query_complexity == QueryComplexity.COMPLEX:
        return {
            "num_query_variations": 5,
            "chunks_per_query": 8,
            "initial_retrieval_k": 14,
            "final_retrieval_k": 12,
            "use_reranking": True,
            "use_domain_knowledge": True,
            "use_self_correction": True,
        }
    else:  # BROAD
        return {
            "num_query_variations": 6,
            "chunks_per_query": 10,
            "initial_retrieval_k": 15,
            "final_retrieval_k": 12,
            "use_reranking": True,
            "use_domain_knowledge": True,
            "use_self_correction": True,
        }


# Domain-specific terminology dictionary
DOMAIN_TERMS = {
    # Product names
    "products": ["Carllm", "Homellm", "Lifellm", "Healthllm", "Bizllm", "Markellm", "Claimllm", "Rellm"],
    # Insurance terminology
    "insurance_terms": [
        "premium", "deductible", "coverage", "policy", "claim", "underwriting", "actuarial",
        "liability", "indemnity", "endorsement", "rider", "exclusion", "beneficiary", "annuity",
        "risk assessment", "loss ratio", "reinsurance", "actuary", "adjuster", "appraisal"
    ],
    # Contract-specific terms
    "contract_terms": [
        "terms and conditions", "policyholder", "insured", "insurer", "effective date", "expiration",
        "renewal", "cancellation", "termination", "grace period", "lapse", "reinstatement",
        "waiver", "subrogation", "arbitration", "dispute resolution"
    ],
    # Product-specific features
    "product_features": [
        "AI-powered", "risk assessment", "instant quoting", "fraud detection", "customer insights",
        "mobile integration", "automated support", "personalization", "analytics", "dashboard"
    ]
}

# Edge case patterns
EDGE_CASE_PATTERNS = [
    r"edge case|rare|unusual|exception|special circumstance",
    r"not covered|excluded|limitation|restriction",
    r"maximum|minimum|limit|cap|threshold",
    r"expir|renewal|cancellation|termination",
    r"dispute|claim denial|appeal|grievance",
    r"pre-existing|waiting period|exclusion period"
]


def extract_domain_terms(question: str) -> List[str]:
    """
    Extract domain-specific terminology from the question.
    Identifies insurance terms, product names, and contract-specific language.
    """
    question_lower = question.lower()
    found_terms = []
    
    # Check for product names
    for product in DOMAIN_TERMS["products"]:
        if product.lower() in question_lower:
            found_terms.append(product)
    
    # Check for insurance terminology
    for term in DOMAIN_TERMS["insurance_terms"]:
        if term in question_lower:
            found_terms.append(term)
    
    # Check for contract terms
    for term in DOMAIN_TERMS["contract_terms"]:
        if term in question_lower:
            found_terms.append(term)
    
    # Check for product features
    for term in DOMAIN_TERMS["product_features"]:
        if term in question_lower:
            found_terms.append(term)
    
    return found_terms


def detect_edge_cases(question: str) -> bool:
    """
    Detect if the question involves edge cases or rare scenarios.
    These require specialized domain knowledge retrieval.
    """
    question_lower = question.lower()
    for pattern in EDGE_CASE_PATTERNS:
        if re.search(pattern, question_lower):
            return True
    return False


def fetch_domain_knowledge(question: str, domain_terms: List[str], is_edge_case: bool) -> List[Document]:
    """Domain-specific knowledge retrieval for insurance terminology and edge cases."""
    if not USE_DOMAIN_KNOWLEDGE:
        return []
    
    domain_docs = []
    
    if domain_terms:
        for term in domain_terms[:3]:
            try:
                term_docs = retriever.invoke(term, k=DOMAIN_KNOWLEDGE_K)
                domain_docs.extend(term_docs)
            except Exception as e:
                logger.warning(f"Domain term retrieval failed for '{term}': {e}")
    
    if is_edge_case:
        try:
            edge_case_query = f"{question} edge case exception rare scenario"
            edge_docs = retriever.invoke(edge_case_query, k=DOMAIN_KNOWLEDGE_K * 2)
            domain_docs.extend(edge_docs)
        except Exception as e:
            logger.warning(f"Edge case retrieval failed: {e}")
    
    for product in DOMAIN_TERMS["products"]:
        if product.lower() in question.lower():
            try:
                product_query = f"{product} features coverage policy terms"
                product_docs = retriever.invoke(product_query, k=DOMAIN_KNOWLEDGE_K)
                domain_docs.extend(product_docs)
            except Exception as e:
                logger.warning(f"Product-specific retrieval failed for '{product}': {e}")
    
    contract_terms_found = [term for term in DOMAIN_TERMS["contract_terms"] if term in question.lower()]
    if contract_terms_found:
        try:
            contract_query = f"{' '.join(contract_terms_found[:2])} contract policy terms conditions"
            contract_docs = retriever.invoke(contract_query, k=DOMAIN_KNOWLEDGE_K)
            domain_docs.extend(contract_docs)
        except Exception as e:
            logger.warning(f"Contract-specific retrieval failed: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in domain_docs:
        doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    return unique_docs


if openrouter_api_key:
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=openrouter_base_url,
        api_key=openrouter_api_key,
        max_tokens=max_tokens,
    )
    query_llm = ChatOpenAI(
        model=MODEL,
        temperature=0.7,
        base_url=openrouter_base_url,
        api_key=openrouter_api_key,
        max_tokens=500,
    )
    critique_llm = ChatOpenAI(
        model=CRITIQUE_MODEL,
        temperature=0,
        base_url=openrouter_base_url,
        api_key=openrouter_api_key,
        max_tokens=max_tokens,
    ) if USE_CRITIQUE_SCORING else None
else:
    llm = ChatOpenAI(temperature=0, model_name=MODEL)
    query_llm = ChatOpenAI(temperature=0.7, model_name=MODEL, max_tokens=500)
    critique_llm = ChatOpenAI(temperature=0, model_name=CRITIQUE_MODEL) if USE_CRITIQUE_SCORING else None


def fetch_context(question: str) -> list[Document]:
    """Adaptive Multi-Query Fusion (RAG-Fusion) with adaptive retrieval parameters."""
    query_complexity = classify_query(question)
    adaptive_params = get_adaptive_parameters(query_complexity)
    keywords = extract_keywords(question)
    
    domain_terms = []
    is_edge_case = False
    if adaptive_params["use_domain_knowledge"]:
        domain_terms = extract_domain_terms(question)
        is_edge_case = detect_edge_cases(question)
    
    if query_complexity == QueryComplexity.SIMPLE or not USE_RAG_FUSION:
        initial_docs = retriever.invoke(question, k=adaptive_params["initial_retrieval_k"])
        filtered_docs = keyword_filter(initial_docs, keywords)
        
        if adaptive_params["use_reranking"] and reranker:
            reranked_docs = rerank_documents(question, filtered_docs)
        else:
            reranked_docs = filtered_docs
        
        if adaptive_params["use_domain_knowledge"]:
            domain_docs = fetch_domain_knowledge(question, domain_terms, is_edge_case)
            all_docs = reranked_docs + domain_docs
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            return unique_docs[:adaptive_params["final_retrieval_k"] + DOMAIN_KNOWLEDGE_K]
        
        return reranked_docs[:adaptive_params["final_retrieval_k"]]
    
    query_variations = generate_query_variations(question, adaptive_params["num_query_variations"])
    all_query_results = []
    for query_var in query_variations:
        query_docs = retriever.invoke(query_var, k=adaptive_params["chunks_per_query"])
        all_query_results.append(query_docs)
    
    merged_docs = reciprocal_rank_fusion(all_query_results)
    filtered_docs = keyword_filter(merged_docs, keywords)
    
    if adaptive_params["use_reranking"] and reranker:
        reranked_docs = rerank_documents(question, filtered_docs)
    else:
        reranked_docs = filtered_docs
    
    if adaptive_params["use_domain_knowledge"]:
        domain_docs = fetch_domain_knowledge(question, domain_terms, is_edge_case)
        all_docs = reranked_docs + domain_docs
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs[:adaptive_params["final_retrieval_k"] + DOMAIN_KNOWLEDGE_K]
    
    return reranked_docs[:adaptive_params["final_retrieval_k"]]


def summarize_chunks(question: str, docs: List[Document]) -> str:
    """Summarize chunks while preserving metadata and completeness."""
    # For questions requiring high completeness, avoid summarization
    question_lower = question.lower()
    requires_high_completeness = any(term in question_lower for term in [
        "all", "complete", "comprehensive", "everything", "list", "all products",
        "tell me about", "overview", "describe"
    ])
    
    # Increase threshold for high-completeness questions
    effective_max_chunks = MAX_CONTEXT_CHUNKS * 2 if requires_high_completeness else MAX_CONTEXT_CHUNKS
    
    if not USE_CONTEXT_SUMMARIZATION or len(docs) <= effective_max_chunks:
        # If summarization disabled or chunks are manageable, return original context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('doc_type', 'Unknown')
            section = doc.metadata.get('section', '')
            
            # Extract filename from source path
            filename = source.split('/')[-1] if '/' in source else source
            
            # Build context with metadata
            metadata_str = f"[Context {i}: {doc_type}/{filename}"
            if section:
                metadata_str += f" - Section: {section}"
            metadata_str += "]"
            
            context_parts.append(f"{metadata_str}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    # Summarize chunks while preserving metadata
    summarized_parts = []
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        doc_type = doc.metadata.get('doc_type', 'Unknown')
        section = doc.metadata.get('section', '')
        filename = source.split('/')[-1] if '/' in source else source
        
        # Create summarization prompt with emphasis on completeness
        summarize_prompt = f"""Summarize the following context chunk while preserving ALL key facts, product names, numbers, dates, and specific details.

Original Context:
{doc.page_content}

**CRITICAL REQUIREMENTS:**
- Preserve ALL factual information (numbers, dates, names, features, specifications) - do NOT omit ANY facts
- Maintain product-specific terminology (Carllm, Homellm, Lifellm, Healthllm, Bizllm, Markellm, Claimllm, Rellm)
- Keep ALL important details relevant to answering questions - completeness is MORE important than brevity
- Include ALL features, use cases, target markets, pricing models, and technical details mentioned
- If the original lists multiple items (features, products, etc.), include ALL of them in the summary
- The summary should be comprehensive - only remove redundant phrasing, not actual information
- Target: 20-30% shorter than original (minimal reduction to preserve completeness)
- If in doubt, include more information rather than less

Summary:"""
        
        try:
            summary_response = llm.invoke([HumanMessage(content=summarize_prompt)])
            summary = summary_response.content.strip()
            
            # Build context with metadata and summary
            metadata_str = f"[Context {i}: {doc_type}/{filename}"
            if section:
                metadata_str += f" - Section: {section}"
            metadata_str += "]"
            
            summarized_parts.append(f"{metadata_str}\n{summary}")
        except Exception as e:
            logger.warning(f"Summarization failed for chunk {i}: {e}")
            # Fallback to original content
            metadata_str = f"[Context {i}: {doc_type}/{filename}"
            if section:
                metadata_str += f" - Section: {section}"
            metadata_str += "]"
            summarized_parts.append(f"{metadata_str}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(summarized_parts)


def verify_and_correct_answer(question: str, initial_answer: str, docs: List[Document]) -> str:
    """Self-correction: cross-check answer against all retrieved chunks."""
    if not USE_SELF_CORRECTION:
        return initial_answer
    
    # Build context from ALL retrieved chunks for thorough verification
    # Use all docs to ensure maximum completeness and accuracy
    full_context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        doc_type = doc.metadata.get('doc_type', 'Unknown')
        section = doc.metadata.get('section', '')
        filename = source.split('/')[-1] if '/' in source else source
        
        metadata_str = f"[Context {i}: {doc_type}/{filename}"
        if section:
            metadata_str += f" - Section: {section}"
        metadata_str += "]"
        
        full_context_parts.append(f"{metadata_str}\n{doc.page_content}")
    
    full_context = "\n\n---\n\n".join(full_context_parts)
    
    # Check if question is about products or company overview
    question_lower = question.lower()
    is_products_question = any(term in question_lower for term in [
        "products", "product suite", "offerings", "all products",
        "tell me about", "overview", "company and"
    ])
    
    # Verification and correction prompt - Streamlined for effectiveness
    verification_prompt = f"""Review this answer against ALL context chunks. Your goal: Accuracy >4.5, Completeness >4.5, Relevance >4.95.

Question: {question}

Initial Answer:
{initial_answer}

All Context Chunks:
{full_context}

**REVIEW PROCESS:**

1. **Check ACCURACY (>4.5) - HIGHEST PRIORITY:**
   - **SYSTEMATIC FACT VERIFICATION**: For EVERY fact in the answer, verify it appears in at least one context chunk
   - **CROSS-REFERENCE ALL CHUNKS**: If a fact appears in multiple chunks, check they are consistent
   - **VERIFY SPECIFIC DETAILS**: Check every number, date, name, product name, feature name, specification against the context
   - **HANDLE CONTRADICTIONS**: If chunks contradict each other, use the most authoritative source or note the conflict
   - **FIX ALL ERRORS**: If ANY fact is incorrect, contradictory, or not in context, you MUST fix it
   - **PRESERVE CORRECT INFO**: Keep all information that is verified as correct
   - **REMOVE UNCERTAIN INFO**: If you cannot verify a fact in the context, remove it rather than risk an error
   - **ZERO ERRORS REQUIRED**: Accuracy >4.5 requires ZERO factual errors. One error drops the score below 4.5.
   - **ACCURACY FIRST**: If there's a conflict between accuracy and completeness, prioritize accuracy

2. **Check COMPLETENESS (>=4.5) - CRITICAL:**
   - **SYSTEMATIC REVIEW REQUIRED**: Process chunks in order (Chunk 1, then Chunk 2, then Chunk 3...)
   - For EACH chunk, ask: "What information does this chunk contain? Is it in the answer?"
   - Create a mental checklist: "Chunk 1 has: [list], Chunk 2 has: [list], Chunk 3 has: [list]..."
   - If ANY chunk contains relevant information NOT in the answer, you MUST add it
   - For list questions, check EVERY chunk and include ALL items found across ALL chunks
   - Missing information from even ONE relevant chunk will drop completeness below 4.5
   - Completeness >=4.5 requires: ALL relevant information from ALL chunks included - NO missing details

3. **Check RELEVANCE (>=4.90) - CRITICAL:**
   - Ensure answer DIRECTLY addresses the specific question asked
   - Remove or minimize information that doesn't directly relate to the question
   - Keep only information that is directly relevant to answering the question
   - Avoid tangents, background info, or context that doesn't directly answer the question
   - Relevance >=4.90 requires: Answer is focused, on-topic, and directly addresses the question
   - If the question asks for specific info (e.g., "features"), focus ONLY on that

**Special Cases:**
- Product questions: Must include all 8 products if asked (Carllm, Homellm, Lifellm, Healthllm, Bizllm, Markellm, Claimllm, Rellm)
- Multi-part questions: Answer ALL parts completely

**OUTPUT:**
- If answer is accurate (>4.5), complete (>=4.5), and relevant (>=4.90): "NO_CHANGES_NEEDED"
- Otherwise, provide corrected answer that:
  * **FIRST: Fixes ALL errors** (for accuracy >4.5) - This is the highest priority
  * **SECOND: Adds ALL missing relevant information** from ALL chunks (for completeness >=4.5)
  * **THIRD: Removes or minimizes unrelated information** (for relevance >=4.90)
  * Preserves all existing correct and relevant information
  * **PRIORITY ORDER**: Accuracy > Completeness > Relevance
  * **VERIFICATION RULE**: Only add information if you can verify it exists in the context chunks
  * Balance: Include all relevant info (completeness) while staying focused (relevance), but NEVER at the cost of accuracy

Review Result:"""
    
    try:
        verification_response = llm.invoke([HumanMessage(content=verification_prompt)])
        corrected_answer = verification_response.content.strip()
        
        # Check if verification said no changes needed
        if "NO_CHANGES_NEEDED" in corrected_answer.upper() or corrected_answer.upper().startswith("NO_CHANGES"):
            return initial_answer
        
        # Safety check: If correction is significantly shorter, it might have lost information
        # But be more lenient - only reject if it's much shorter (50% threshold)
        if len(corrected_answer) < len(initial_answer) * 0.5:
            # Correction seems too short, might be incomplete - return original
            logger.warning(f"Verification result too short ({len(corrected_answer)} vs {len(initial_answer)}), keeping original answer")
            return initial_answer
        
        # Accept longer corrections - they likely added missing information for completeness
        # But verify it's not introducing errors (accuracy check)
        if len(corrected_answer) > len(initial_answer) * 1.3:
            # Longer correction - likely added missing information, accept it
            logger.info(f"Verification result longer ({len(corrected_answer)} vs {len(initial_answer)}), accepting correction (likely improved completeness)")
        
        # Final accuracy safeguard: If correction removed critical verified facts, prefer original
        # Check if key facts from original are still present (basic sanity check)
        key_terms_original = set(re.findall(r'\b(Carllm|Homellm|Lifellm|Healthllm|Bizllm|Markellm|Claimllm|Rellm|Insurellm|2015|32)\b', initial_answer, re.IGNORECASE))
        key_terms_corrected = set(re.findall(r'\b(Carllm|Homellm|Lifellm|Healthllm|Bizllm|Markellm|Claimllm|Rellm|Insurellm|2015|32)\b', corrected_answer, re.IGNORECASE))
        
        # If correction removed many key verified facts, it might have introduced errors
        if len(key_terms_original) > 3 and len(key_terms_corrected) < len(key_terms_original) * 0.7:
            logger.warning(f"Correction removed many key facts ({len(key_terms_corrected)} vs {len(key_terms_original)}), keeping original for accuracy")
            return initial_answer
        
        return corrected_answer
    except Exception as e:
        logger.warning(f"Self-correction failed: {e}")
        return initial_answer


def score_answer(question: str, answer: str, docs: List[Document]) -> Dict[str, float]:
    """
    Optional: Use a second LLM (critique model) to score the answer before returning it.
    
    Returns scores for accuracy, completeness, and relevance (0-5 scale).
    """
    if not USE_CRITIQUE_SCORING or not critique_llm:
        return {}
    
    # Build context summary for scoring
    context_summary = "\n\n".join([
        f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content[:200]}..."
        for doc in docs[:5]  # Use top 5 chunks for scoring
    ])
    
    scoring_prompt = f"""Evaluate the following answer on three metrics (0-5 scale):

Question: {question}

Answer:
{answer}

Relevant Context:
{context_summary}

Provide scores for:
1. Accuracy (0-5): Is the answer factually correct based on the context? No hallucinations?
2. Completeness (0-5): Does the answer include all relevant facts from the context?
3. Relevance (0-5): Does the answer directly address the question?

Respond in JSON format:
{{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>
}}"""
    
    try:
        score_response = critique_llm.invoke([HumanMessage(content=scoring_prompt)])
        score_text = score_response.content.strip()
        
        # Extract JSON from response
        # Try to find JSON in the response
        json_start = score_text.find('{')
        json_end = score_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            score_json = json.loads(score_text[json_start:json_end])
            return score_json
    except Exception as e:
        logger.warning(f"Answer scoring failed: {e}")
    
    return {}


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """Answer the question using Adaptive RAG with conditional self-correction."""
    query_complexity = classify_query(question)
    adaptive_params = get_adaptive_parameters(query_complexity)
    docs = fetch_context(question)
    
    total_chars = sum(len(doc.page_content) for doc in docs)
    estimated_tokens = total_chars / 4
    context_limit = max_tokens * 0.8
    
    if USE_CONTEXT_SUMMARIZATION and estimated_tokens > context_limit:
        context = summarize_chunks(question, docs)
    else:
        context = "\n\n".join([
            f"[Context {i+1} from {doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
    
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    initial_response = llm.invoke(messages)
    initial_answer = initial_response.content
    
    if adaptive_params["use_self_correction"] and USE_SELF_CORRECTION:
        corrected_answer = verify_and_correct_answer(question, initial_answer, docs)
        if adaptive_params["use_self_correction"] and USE_CRITIQUE_SCORING:
            scores = score_answer(question, corrected_answer, docs)
            if scores:
                logger.info(f"Answer Scores - Accuracy: {scores.get('accuracy', 'N/A')}/5, "
                           f"Completeness: {scores.get('completeness', 'N/A')}/5, "
                           f"Relevance: {scores.get('relevance', 'N/A')}/5")
        return corrected_answer, docs
    else:
        return initial_answer, docs
