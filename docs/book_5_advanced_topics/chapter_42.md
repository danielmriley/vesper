# Vesper Future Plans - Chapter 42: Native C++ Tokenizer

## 1. Goal

Implement a Byte-Pair Encoding (BPE) tokenizer in C++. This removes the dependency on Python tokenizers (like HuggingFace `tokenizers`) for inference applications.

## 2. Features

-   **Training (Optional):** Implementing BPE training is complex; we might skip this or implement a simple version.
-   **Inference (Encoding):** The critical part. Given a string, convert it to a sequence of integer tokens based on a vocabulary file (e.g., `vocab.json` and `merges.txt` from GPT-2/RoBERTa).
-   **Inference (Decoding):** Convert token IDs back to a string.
-   **Efficiency:** Use a Trie or Hash Map for efficient merge lookups.

## 3. Why It's Next

For Vesper to be a standalone inference engine for LLMs (e.g., embedded in a game engine or mobile app), it needs to handle raw text input. Calling out to Python just to tokenize text destroys performance and portability.
