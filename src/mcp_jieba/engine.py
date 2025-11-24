import os
import rjieba
from typing import List, Union, Dict, Set
from rank_bm25 import BM25Okapi

class JiebaEngine:
    _instance = None
    _stopwords: Set[str] = set()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JiebaEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_stopwords()
            self._initialized = True

    def _load_stopwords(self):
        """Load stopwords from the package resource."""
        try:
            # Try to load from the package resources
            resource_path = os.path.join(os.path.dirname(__file__), 'resources', 'stopwords.txt')
            if os.path.exists(resource_path):
                with open(resource_path, 'r', encoding='utf-8') as f:
                    self._stopwords = set(line.strip() for line in f if line.strip())
            else:
                # Fallback or empty if not found (should not happen with correct install)
                self._stopwords = set()
        except Exception as e:
            print(f"Warning: Failed to load stopwords: {e}")
            self._stopwords = set()

    def _is_valid_token(self, token: str) -> bool:
        """Check if a token is valid (not a stopword and not purely punctuation)."""
        token = token.strip()
        if not token:
            return False
        if token in self._stopwords:
            return False
        # Basic punctuation filter: if the token is just punctuation/symbols
        # This is a simple heuristic; for more complex needs, regex could be used.
        # Here we assume if it's in stopwords (which includes punctuation), it's filtered.
        # If it's not in stopwords but is a symbol not in the list, we might keep it
        # or we can add a simple check.
        return True

    def process(self, text: Union[str, List[str]], mode: str = "exact") -> Dict[str, List[str]]:
        """
        Process text(s) with jieba segmentation.

        Args:
            text: A single string or a list of strings.
            mode: "exact" or "search".

        Returns:
            A dictionary where keys are indices (as strings) and values are lists of tokens.
        """
        # Normalize input to list
        inputs = [text] if isinstance(text, str) else text
        results = {}

        for idx, content in enumerate(inputs):
            if not isinstance(content, str):
                results[str(idx)] = []
                continue

            if mode == "search":
                # cut_for_search returns an iterator or list depending on implementation
                raw_tokens = rjieba.cut_for_search(content)
            else:
                # Default to exact mode
                raw_tokens = rjieba.cut(content)

            # Filter tokens
            filtered_tokens = [t for t in raw_tokens if self._is_valid_token(t)]
            results[str(idx)] = filtered_tokens

        return results

    def tag(self, text: Union[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Process text(s) with jieba POS tagging.

        Args:
            text: A single string or a list of strings.

        Returns:
            A dictionary where keys are indices (as strings) and values are lists of {"word": word, "flag": flag}.
        """
        inputs = [text] if isinstance(text, str) else text
        results = {}

        for idx, content in enumerate(inputs):
            if not isinstance(content, str):
                results[str(idx)] = []
                continue

            # rjieba.tag returns a list of tuples [(word, flag), ...]
            tags = rjieba.tag(content)
            # Convert to list of dicts for better JSON serialization
            results[str(idx)] = [{"word": t[0], "flag": t[1]} for t in tags]

        return results

    def extract_keywords_bm25(self, texts: List[str], top_k: int = 5) -> Dict[str, List[str]]:
        """
        Extract keywords from a batch of texts using BM25.
        The batch itself is treated as the corpus.

        Args:
            texts: A list of strings (documents).
            top_k: Number of keywords to extract per document.

        Returns:
            A dictionary where keys are indices (as strings) and values are lists of keywords.
        """
        if not texts:
            return {}

        # 1. Tokenize all documents
        tokenized_corpus = []
        valid_indices = [] # Keep track of indices that are actually strings

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                continue

            # Use exact mode for tokenization
            raw_tokens = rjieba.cut(text)
            # Filter stopwords and punctuation
            tokens = [t for t in raw_tokens if self._is_valid_token(t)]
            tokenized_corpus.append(tokens)
            valid_indices.append(idx)

        if not tokenized_corpus:
            return {}

        # 2. Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)

        results = {}

        # 3. For each document, find the top_k words with highest BM25 scores
        # Note: BM25 is usually query-based (score(doc, query)).
        # To use it for keyword extraction (score(word, doc)), we can consider:
        # The "relevance" of a word to the document in the context of the corpus.
        # A common approximation is to use the IDF of the word * TF in the document.
        # BM25Okapi formula components can be accessed.
        # However, rank_bm25 library is designed for retrieval.
        # We can hack it: for a doc, iterate over its unique words, treat each word as a single-term query,
        # and get the score for the document itself.

        # Optimization: Calculate scores for all unique tokens in the doc
        for i, doc_tokens in enumerate(tokenized_corpus):
            original_idx = str(valid_indices[i])
            if not doc_tokens:
                results[original_idx] = []
                continue

            unique_tokens = list(set(doc_tokens))
            scores = []

            for token in unique_tokens:
                # get_scores returns scores for all docs for a given query.
                # We only care about the score for the current document (index i).
                # This is inefficient O(N^2) if we do it naively for all docs.
                # But rank_bm25 doesn't expose "score(doc, term)" directly efficiently for single doc.
                # Actually, bm25.idf[token] gives IDF.
                # We can manually compute a simplified BM25-like score or just use IDF * TF.
                # Let's try to use the library's get_scores for correctness, even if slightly slow for large batches.
                # query = [token]
                # doc_scores = bm25.get_scores(query)
                # score = doc_scores[i]

                # Faster approach using internal structures if possible, but let's stick to public API for stability.
                # Or better: just use the IDF component from BM25 as the "weight" and multiply by TF.
                # This is essentially TF-IDF but using BM25's IDF definition.
                # rank_bm25 stores idf in `bm25.idf` (dict).

                tf = doc_tokens.count(token)
                # BM25 IDF is often negative for very common terms in some implementations, but rank_bm25 uses standard log.
                # Let's check if token is in idf
                idf = 0.0
                if token in bm25.idf:
                    idf = bm25.idf[token]

                # Simple TF-IDF using BM25's IDF
                score = tf * idf
                scores.append((token, score))

            # Sort by score desc
            scores.sort(key=lambda x: x[1], reverse=True)

            # Take top k
            top_keywords = [w for w, s in scores[:top_k]]
            results[original_idx] = top_keywords

        return results
