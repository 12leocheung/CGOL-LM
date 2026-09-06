import os
import pickle
import random
import numpy as np

d = 10000
model_filename = "cgol_model.pkl"

# Global references loaded from model
board = None
word_groups = {}
sentences = []
words = {}
semantic_words = {}


class Grid:

    def __init__(self, width: int, height: int, default_value="."):
        self.width = width
        self.height = height
        self._grid = [
            [default_value for _ in range(width)] for _ in range(height)
        ]

    def _is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def set(self, x: int, y: int, value):
        if not self._is_valid(x, y):
            raise IndexError(f"Position ({x}, {y}) is out of grid bounds.")
        self._grid[y][x] = value

    def get(self, x: int, y: int):
        if not self._is_valid(x, y):
            raise IndexError(f"Position ({x}, {y}) is out of grid bounds.")
        return self._grid[y][x]

    def get_info(self, target_word: str):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.get(x, y)
                if isinstance(cell, dict) and cell.get("content") == target_word:
                    status = cell.get("status", "dead")
                    return x, y, status
        return None

    def display(self):
        str_grid = []
        for row in self._grid:
            str_row = []
            for cell in row:
                if isinstance(cell, dict) and "content" in cell:
                    str_row.append(str(cell["content"]))
                else:
                    str_row.append(str(cell))
            str_grid.append(str_row)

        col_width = (
            max(len(s) for row in str_grid for s in row) if str_grid else 0
        )

        for row in str_grid:
            formatted_row = [s.ljust(col_width) for s in row]
            print(" | ".join(formatted_row))


def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / d)


def load_model(filename):
    global board, word_groups, sentences, words, semantic_words
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Model file '{filename}' not found. Please run CGOL_train.py first to generate it."
        )

    with open(filename, "rb") as f:
        state = pickle.load(f)

    board = state["board"]
    word_groups = state["word_groups"]
    sentences = state["sentences"]
    words = state["words"]
    semantic_words = state["semantic_words"]
    print(f"Successfully loaded model from '{filename}'.")


def generate_sentence(seed_phrase: str, max_length: int = 15) -> str:
    raw_words = seed_phrase.strip().split()
    if not raw_words:
        return ""

    generated_sequence = []

    for word in raw_words:
        clean = word.strip(".,!?")
        matched = None
        for variant in [word, clean, clean.lower(), clean.capitalize()]:
            if variant in words:
                matched = variant
                break
        if matched:
            generated_sequence.append(matched)

    if not generated_sequence:
        print(f"Error: None of the words in '{seed_phrase}' are in vocabulary.")
        return ""

    current_word = generated_sequence[-1]
    seen_bigrams = set()

    for i in range(len(generated_sequence) - 1):
        seen_bigrams.add((generated_sequence[i], generated_sequence[i + 1]))

    for _ in range(max_length - len(generated_sequence)):
        if current_word.endswith((".", "?", "!")) and len(generated_sequence) > 1:
            break

        followers = set()
        for sentence in sentences:
            for i in range(len(sentence) - 1):
                if sentence[i] == current_word:
                    followers.add(sentence[i + 1])

        if not followers:
            info = board.get_info(current_word)
            if info:
                tx, ty, _ = info
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = tx + dx, ty + dy
                        if board._is_valid(nx, ny):
                            cell = board.get(nx, ny)
                            if isinstance(cell, dict) and "content" in cell:
                                followers.add(cell["content"])

        if not followers:
            break

        candidates = [
            f
            for f in followers
            if (current_word, f) not in seen_bigrams
            and f not in generated_sequence[-3:]
        ]
        if not candidates:
            candidates = [
                f for f in followers if (current_word, f) not in seen_bigrams
            ]
        if not candidates:
            candidates = list(followers)

        lookup_key = current_word.strip(".,!?")
        if lookup_key not in semantic_words:
            lookup_key = current_word

        target_vec = semantic_words[lookup_key]

        scored_candidates = []
        for cand in candidates:
            cand_key = cand.strip(".,!?")
            c_vec = semantic_words.get(
                cand_key, semantic_words.get(cand, target_vec)
            )
            sim = similarity(target_vec, c_vec)
            scored_candidates.append((cand, sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        top_k = scored_candidates[:3]
        best_next_word = random.choice(top_k)[0]

        seen_bigrams.add((current_word, best_next_word))
        generated_sequence.append(best_next_word)
        current_word = best_next_word

    return " ".join(generated_sequence)


if __name__ == "__main__":
    load_model(model_filename)
    board.display()

    while True:
        seed = input(
            "\nEnter a seed word to generate a sentence (or 'exit' to quit): "
        ).strip()
        if seed.lower() == "exit":
            break
        output = generate_sentence(seed, max_length=15)
        print(f"Generated Sequence: {output}")