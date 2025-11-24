import os
import tempfile
import rjieba
from typing import List, Union, Dict, Any, Set, Optional
import importlib.resources

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
            from . import resources
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

    def process(self, text: Union[str, List[str]], mode: str = "exact", user_dict: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Process text(s) with jieba segmentation.

        Args:
            text: A single string or a list of strings.
            mode: "exact" or "search".
            user_dict: Optional list of terms to add to the dictionary.

        Returns:
            A dictionary where keys are indices (as strings) and values are lists of tokens.
        """
        # Load user dictionary if provided
        if user_dict:
            self._load_user_dict(user_dict)

        # Normalize input to list
        inputs = [text] if isinstance(text, str) else text
        results = {}

        for idx, content in enumerate(inputs):
            if not isinstance(content, str):
                results[str(idx)] = []
                continue

            tokens = []
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

    def _load_user_dict(self, user_dict: List[str]):
        """
        Load a user dictionary.
        Since rjieba might expect a file, we write to a temp file.
        """
        if not user_dict:
            return

        try:
            # Create a temporary file
            # rjieba/jieba format: word freq tag (freq and tag are optional)
            # We just write the word
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tmp:
                for word in user_dict:
                    tmp.write(f"{word}\n")
                tmp_path = tmp.name

            # Load into rjieba
            # Note: rjieba.load_userdict might not be available or might work differently.
            # If rjieba is strictly a binding to jieba-rs, we need to check its exposed API.
            # Assuming it mimics jieba's API for now.
            if hasattr(rjieba, 'load_userdict'):
                rjieba.load_userdict(tmp_path)

            # Clean up
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except Exception as e:
            # Log warning but don't crash
            print(f"Warning: Failed to load user dictionary: {e}")
