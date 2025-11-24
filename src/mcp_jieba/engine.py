import os
import re
from numpy.random import f
import rjieba
import numpy as np
from typing import List, Union, Dict, Set


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
            resource_path = os.path.join(
                os.path.dirname(__file__), "resources", "stopwords.txt"
            )
            if os.path.exists(resource_path):
                with open(resource_path, "r", encoding="utf-8") as f:
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

    def process(
        self, text: Union[str, List[str]], mode: str = "exact"
    ) -> Dict[int, List[str]]:
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
                results[idx] = []
                continue

            if mode == "search":
                # cut_for_search returns an iterator or list depending on implementation
                raw_tokens = rjieba.cut_for_search(content)
            else:
                # Default to exact mode
                raw_tokens = rjieba.cut(content)

            # Filter tokens
            filtered_tokens = [t for t in raw_tokens if self._is_valid_token(t)]
            results[idx] = filtered_tokens

        return results

    def tag(self, text: Union[str, List[str]]) -> Dict[int, Dict[str, str]]:
        """
        Process text(s) with jieba POS tagging.

        Args:
            text: A single string or a list of strings.

        Returns:
            A dictionary where keys are indices (as strings) and values are dicts of {word : flag}.
        """
        inputs = [text] if isinstance(text, str) else text
        results = {}

        for idx, content in enumerate(inputs):
            if not isinstance(content, str):
                results[idx] = {}
                continue

            # rjieba.tag returns a list of tuples [(word, flag), ...]
            tags = rjieba.tag(content)
            # Convert to dict for better JSON serialization
            results[idx]= {f"{t[0]}": f"{t[1]}" for t in tags}

        return results

    def extract_keywords_bm25(
        self, texts: Union[str, List[str]], top_k: int = 5
    ) -> Dict[int, List[str]]:
        """
        Extract keywords from text(s) using BM25-adpt (Numpy implementation).
        Each input string is treated as a corpus (split into sentences).

        Args:
            texts: A single string or a list of strings.
            top_k: Number of keywords to extract per document.

        Returns:
            A dictionary where keys are indices (as strings) and values are lists of keywords.
            Even for a single string input, returns {'0': [...]}.
        """
        # Normalize input to list
        is_single_input = isinstance(texts, str)
        inputs = [texts] if is_single_input else texts

        results = {}

        for idx, text in enumerate(inputs):
            if not isinstance(text, str) or not text.strip():
                results[idx] = []
                continue

            # 1. Split into sentences to form the corpus
            # Simple split by common Chinese/English sentence terminators
            sentences = re.split(r"[。！？!?\n;；]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                results[idx] = []
                continue

            # 2. Tokenize sentences
            vocab = {}
            next_id = 0

            # We need to map tokens to indices for numpy
            docs_tokens_ids = []

            for sent in sentences:
                raw_tokens = rjieba.cut(sent)
                tokens = [t for t in raw_tokens if self._is_valid_token(t)]
                if not tokens:
                    continue

                token_ids = []
                for t in tokens:
                    if t not in vocab:
                        vocab[t] = next_id
                        next_id += 1
                    token_ids.append(vocab[t])

                docs_tokens_ids.append(token_ids)

            if not docs_tokens_ids:
                results[idx] = []
                continue

            # 3. Build Numpy Matrices
            N = len(docs_tokens_ids)
            V = len(vocab)

            # TF Matrix
            tf_matrix = np.zeros((N, V), dtype=np.float32)
            for i, doc_ids in enumerate(docs_tokens_ids):
                for tid in doc_ids:
                    tf_matrix[i, tid] += 1

            # 4. Compute BM25 Components
            # k1 = 1.5, b = 0.75 (standard defaults)
            k1 = 1.5
            b = 0.75

            # Document Frequencies (number of docs containing term)
            df = np.count_nonzero(tf_matrix, axis=0)

            # IDF
            # standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

            # Document Lengths
            doc_lens = tf_matrix.sum(axis=1)
            avgdl = doc_lens.mean() if N > 0 else 0

            if avgdl == 0:
                results[idx] = []
                continue

            # BM25 Scores
            # score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

            # Reshape for broadcasting
            # tf_matrix: (N, V)
            # idf: (V,) -> broadcast to (N, V)
            # doc_lens: (N,) -> broadcast to (N, 1)

            numerator = tf_matrix * (k1 + 1)

            # Denominator term: k1 * (1 - b + b * doc_lens[:, None] / avgdl)
            # Shape (N, 1)
            denom_term = k1 * (1 - b + b * doc_lens[:, None] / avgdl)

            denominator = tf_matrix + denom_term

            # Avoid division by zero (shouldn't happen as denom_term > 0 usually, but safe check)
            # scores_matrix (N, V)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores_matrix = idf * (numerator / denominator)
                scores_matrix = np.nan_to_num(scores_matrix)

            # 5. Aggregate scores for the whole text
            # Sum scores across all sentences for each word
            word_scores = scores_matrix.sum(axis=0)

            # 6. Get Top-K
            # Get indices of top-k scores
            if V <= top_k:
                top_indices = np.argsort(word_scores)[::-1]
            else:
                # argpartition is faster for large V
                top_indices = np.argpartition(word_scores, -top_k)[-top_k:]
                # Sort the top k
                top_indices = top_indices[np.argsort(word_scores[top_indices])[::-1]]

            # Map back to words
            inv_vocab = {v: k for k, v in vocab.items()}
            top_keywords = [inv_vocab[i] for i in top_indices]

            results[idx] = top_keywords

        return results
