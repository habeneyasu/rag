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

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_DIR / "rag.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(file_handler)
logger.propagate = False
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

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
max_tokens = int(os.getenv("MAX_TOKENS", "2000"))

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


def extract_keywords(question: str) -> List[str]:
    """Extract keywords from the question for filtering."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who", "when", "where", "how", "why", "which", "about", "for", "with", "from"}
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def keyword_filter(docs: List[Document], keywords: List[str]) -> List[Document]:
    """Filter documents by keyword presence."""
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
    
    target_variations = num_variations if num_variations is not None else NUM_QUERY_VARIATIONS
    
    if target_variations == 1:
        return [original_query]
    
    prompt = f"""Generate {target_variations} diverse query variations for the following question about Insurellm.
Each variation should use different wording, include relevant synonyms, and maintain the core intent.

Original question: {original_query}

Return ONLY a numbered list of {target_variations} query variations, one per line:"""
    
    try:
        response = query_llm.invoke([HumanMessage(content=prompt)])
        variations_text = response.content.strip()
        
        variations = []
        for line in variations_text.split('\n'):
            line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if line and len(line) > 10:
                variations.append(line)
        
        if original_query not in variations:
            variations.insert(0, original_query)
        
        while len(variations) < target_variations:
            variations.append(original_query)
        
        return variations[:target_variations]
    
    except Exception as e:
        logger.warning(f"Query variation generation failed: {e}")
        return [original_query]


def reciprocal_rank_fusion(all_results: List[List[Document]]) -> List[Document]:
    """Merge multiple retrieval results using Reciprocal Rank Fusion (RRF)."""
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    
    for query_results in all_results:
        for rank, doc in enumerate(query_results, start=1):
            doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
            doc_map[doc_id] = doc
            doc_scores[doc_id] += 1.0 / (RRF_K + rank)
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def rerank_documents(question: str, docs: List[Document], max_rerank: int = 10) -> List[Document]:
    """Rerank documents using cross-encoder for better relevance."""
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
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    BROAD = "broad"


def classify_query(question: str) -> QueryComplexity:
    """Classify query complexity using rule-based heuristics."""
    question_lower = question.lower()
    
    broad_indicators = [
        "all products", "products", "product suite", "offerings", "tell me about",
        "overview", "company and", "insurellm and", "everything", "complete",
        "comprehensive", "list all", "what are all"
    ]
    if any(indicator in question_lower for indicator in broad_indicators):
        return QueryComplexity.BROAD
    
    complex_indicators = [
        " and ", " or ", " both ", " also ", " as well as ",
        "how does", "how do", "explain", "describe", "compare",
        "contract", "pricing", "cost", "tier", "plan",
        "edge case", "exception", "special", "unusual"
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
    """Extract domain-specific terminology from the question."""
    question_lower = question.lower()
    found_terms = []
    
    for category in ["products", "insurance_terms", "contract_terms", "product_features"]:
        for term in DOMAIN_TERMS[category]:
            if term.lower() in question_lower:
                found_terms.append(term)
    
    return found_terms


def detect_edge_cases(question: str) -> bool:
    """Detect if the question involves edge cases or rare scenarios."""
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in EDGE_CASE_PATTERNS)


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents while preserving order."""
    seen = set()
    unique_docs = []
    for doc in docs:
        doc_id = f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content[:100])}"
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    return unique_docs


def fetch_domain_knowledge(question: str, domain_terms: List[str], is_edge_case: bool) -> List[Document]:
    """Domain-specific knowledge retrieval for insurance terminology and edge cases."""
    if not USE_DOMAIN_KNOWLEDGE:
        return []
    
    domain_docs = []
    question_lower = question.lower()
    
    if domain_terms:
        for term in domain_terms[:3]:
            try:
                domain_docs.extend(retriever.invoke(term, k=DOMAIN_KNOWLEDGE_K))
            except Exception as e:
                logger.warning(f"Domain term retrieval failed for '{term}': {e}")
    
    if is_edge_case:
        try:
            edge_docs = retriever.invoke(f"{question} edge case exception rare scenario", k=DOMAIN_KNOWLEDGE_K * 2)
            domain_docs.extend(edge_docs)
        except Exception as e:
            logger.warning(f"Edge case retrieval failed: {e}")
    
    for product in DOMAIN_TERMS["products"]:
        if product.lower() in question_lower:
            try:
                product_docs = retriever.invoke(f"{product} features coverage policy terms", k=DOMAIN_KNOWLEDGE_K)
                domain_docs.extend(product_docs)
            except Exception as e:
                logger.warning(f"Product-specific retrieval failed for '{product}': {e}")
    
    contract_terms_found = [term for term in DOMAIN_TERMS["contract_terms"] if term in question_lower]
    if contract_terms_found:
        try:
            contract_docs = retriever.invoke(
                f"{' '.join(contract_terms_found[:2])} contract policy terms conditions", k=DOMAIN_KNOWLEDGE_K
            )
            domain_docs.extend(contract_docs)
        except Exception as e:
            logger.warning(f"Contract-specific retrieval failed: {e}")
    
    return _deduplicate_docs(domain_docs)


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
            all_docs = _deduplicate_docs(reranked_docs + domain_docs)
            return all_docs[:adaptive_params["final_retrieval_k"] + DOMAIN_KNOWLEDGE_K]
        
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
        all_docs = _deduplicate_docs(reranked_docs + domain_docs)
        return all_docs[:adaptive_params["final_retrieval_k"] + DOMAIN_KNOWLEDGE_K]
    
    return reranked_docs[:adaptive_params["final_retrieval_k"]]


def _format_doc_metadata(doc: Document, index: int) -> str:
    """Format document metadata for context display."""
    source = doc.metadata.get('source', 'Unknown')
    doc_type = doc.metadata.get('doc_type', 'Unknown')
    section = doc.metadata.get('section', '')
    filename = source.split('/')[-1] if '/' in source else source
    
    metadata_str = f"[Context {index}: {doc_type}/{filename}"
    if section:
        metadata_str += f" - Section: {section}"
    metadata_str += "]"
    return metadata_str


def summarize_chunks(question: str, docs: List[Document]) -> str:
    """Summarize chunks while preserving metadata and completeness."""
    question_lower = question.lower()
    requires_high_completeness = any(term in question_lower for term in [
        "all", "complete", "comprehensive", "everything", "list", "all products",
        "tell me about", "overview", "describe"
    ])
    
    effective_max_chunks = MAX_CONTEXT_CHUNKS * 2 if requires_high_completeness else MAX_CONTEXT_CHUNKS
    
    if not USE_CONTEXT_SUMMARIZATION or len(docs) <= effective_max_chunks:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata_str = _format_doc_metadata(doc, i)
            context_parts.append(f"{metadata_str}\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)
    
    summarized_parts = []
    for i, doc in enumerate(docs, 1):
        summarize_prompt = f"""Summarize this context chunk while preserving ALL key facts, numbers, dates, and product names.
Target: 20-30% shorter, but keep all factual information.

Original Context:
{doc.page_content}

Summary:"""
        
        try:
            summary_response = llm.invoke([HumanMessage(content=summarize_prompt)])
            summary = summary_response.content.strip()
            metadata_str = _format_doc_metadata(doc, i)
            summarized_parts.append(f"{metadata_str}\n{summary}")
        except Exception as e:
            logger.warning(f"Summarization failed for chunk {i}: {e}")
            metadata_str = _format_doc_metadata(doc, i)
            summarized_parts.append(f"{metadata_str}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(summarized_parts)


def verify_and_correct_answer(question: str, initial_answer: str, docs: List[Document]) -> str:
    """Self-correction: cross-check answer against all retrieved chunks."""
    if not USE_SELF_CORRECTION:
        return initial_answer
    
    full_context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata_str = _format_doc_metadata(doc, i)
        full_context_parts.append(f"{metadata_str}\n{doc.page_content}")
    
    full_context = "\n\n---\n\n".join(full_context_parts)
    
    verification_prompt = f"""Review this answer against all context chunks. Goal: Accuracy >4.5, Completeness >4.5, Relevance >4.95.

Question: {question}
Initial Answer: {initial_answer}
All Context Chunks: {full_context}

Check:
1. ACCURACY (>4.5): Verify every fact appears in context. Fix any errors. Remove unverified facts.
2. COMPLETENESS (>=4.5): Process each chunk systematically. Add all missing relevant information.
3. RELEVANCE (>=4.90): Ensure answer directly addresses the question. Remove unrelated content.

Priority: Accuracy > Completeness > Relevance

If answer meets all criteria, respond "NO_CHANGES_NEEDED". Otherwise, provide corrected answer:"""
    
    try:
        verification_response = llm.invoke([HumanMessage(content=verification_prompt)])
        corrected_answer = verification_response.content.strip()
        
        if "NO_CHANGES_NEEDED" in corrected_answer.upper() or corrected_answer.upper().startswith("NO_CHANGES"):
            return initial_answer
        
        if len(corrected_answer) < len(initial_answer) * 0.5:
            logger.warning(f"Verification result too short, keeping original")
            return initial_answer
        
        if len(corrected_answer) > len(initial_answer) * 1.3:
            logger.info(f"Verification result longer, accepting correction")
        
        key_terms_original = set(re.findall(
            r'\b(Carllm|Homellm|Lifellm|Healthllm|Bizllm|Markellm|Claimllm|Rellm|Insurellm|2015|32)\b',
            initial_answer, re.IGNORECASE
        ))
        key_terms_corrected = set(re.findall(
            r'\b(Carllm|Homellm|Lifellm|Healthllm|Bizllm|Markellm|Claimllm|Rellm|Insurellm|2015|32)\b',
            corrected_answer, re.IGNORECASE
        ))
        
        if len(key_terms_original) > 3 and len(key_terms_corrected) < len(key_terms_original) * 0.7:
            logger.warning(f"Correction removed many key facts, keeping original")
            return initial_answer
        
        return corrected_answer
    except Exception as e:
        logger.warning(f"Self-correction failed: {e}")
        return initial_answer


def score_answer(question: str, answer: str, docs: List[Document]) -> Dict[str, float]:
    """Score answer quality using LLM critique (0-5 scale for accuracy, completeness, relevance)."""
    if not USE_CRITIQUE_SCORING or not critique_llm:
        return {}
    
    context_summary = "\n\n".join([
        f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content[:200]}..."
        for doc in docs[:5]
    ])
    
    scoring_prompt = f"""Evaluate this answer (0-5 scale for each metric):

Question: {question}
Answer: {answer}
Context: {context_summary}

Provide JSON: {{"accuracy": <score>, "completeness": <score>, "relevance": <score>}}"""
    
    try:
        score_response = critique_llm.invoke([HumanMessage(content=scoring_prompt)])
        score_text = score_response.content.strip()
        
        json_start = score_text.find('{')
        json_end = score_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(score_text[json_start:json_end])
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
        if USE_CRITIQUE_SCORING:
            scores = score_answer(question, corrected_answer, docs)
            if scores:
                logger.info(f"Answer Scores - Accuracy: {scores.get('accuracy', 'N/A')}/5, "
                           f"Completeness: {scores.get('completeness', 'N/A')}/5, "
                           f"Relevance: {scores.get('relevance', 'N/A')}/5")
        return corrected_answer, docs
    
    return initial_answer, docs
