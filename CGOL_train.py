import json
import math
import os
import pickle
import random
import numpy as np

d = 10000
window = 2
similarity_threshold = 0.04
file_path = r"medium_dataset.txt"
model_filename = "cgol_model.pkl"

words = {}
semantic_words = {}
word_groups = {}
sentences = []


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


# --- HDC Primitives ---
def shift(vec: np.ndarray, offset: int) -> np.ndarray:
    return np.roll(vec, offset)


def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 * v2


def bundle(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.random.choice([-1, 1], size=d)
    summed = np.sum(vectors, axis=0)
    res = np.sign(summed)
    res[res == 0] = 1
    return res


def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / d)


def setup():
    global words, semantic_words, word_groups, sentences
    words = {}
    semantic_words = {}
    word_groups = {}

    with open(file_path, "r") as f:
        sentences = json.load(f)

    unique_words = []
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words[word] = np.random.choice([-1, 1], size=d)
                unique_words.append(word)

    word_context_histories = {w: [] for w in unique_words}

    for sentence in sentences:
        seq_len = len(sentence)
        for i, target_word in enumerate(sentence):
            bound_context_components = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                ctx_idx = i + offset
                if 0 <= ctx_idx < seq_len:
                    ctx_word = sentence[ctx_idx]
                    bound_context_components.append(
                        shift(words[ctx_word], offset)
                    )

            if bound_context_components:
                ctx_rep = bundle(bound_context_components)
                word_context_histories[target_word].append(ctx_rep)

    for w in unique_words:
        if word_context_histories[w]:
            semantic_words[w] = bundle(word_context_histories[w])
        else:
            semantic_words[w] = words[w]

    group_vectors = {}
    word_groups = {}

    for w in unique_words:
        vec = semantic_words[w]
        best_gid = None
        best_sim = -1.0

        for gid, grp_vec in group_vectors.items():
            sim = similarity(vec, grp_vec)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= similarity_threshold:
            group_vectors[best_gid] = bundle([group_vectors[best_gid], vec])
            word_groups[best_gid].append(w)
        else:
            new_id = len(group_vectors)
            group_vectors[new_id] = vec
            word_groups[new_id] = [w]

    buffered_count = len(unique_words) + 2
    width = math.ceil(math.sqrt(buffered_count))
    height = math.ceil(buffered_count / width)

    board = Grid(width=width, height=height, default_value=".")

    all_positions = [(x, y) for x in range(width) for y in range(height)]
    random.shuffle(all_positions)

    for word in unique_words:
        rx, ry = all_positions.pop()
        board.set(
            rx, ry, {"content": word, "value": words[word], "status": "dead"}
        )

    print(
        f"Created {len(word_groups)} distinct word groups from {len(unique_words)} words."
    )
    return board, word_groups, sentences


def train(board, word_groups, sentences, words, epochs=25):
    print(f"\n--- Starting Training ({epochs} Epochs) ---")
    next_words = {}
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            curr, nxt = sentence[i], sentence[i + 1]
            next_words.setdefault(curr, set()).add(nxt)

    for epoch in range(1, epochs + 1):
        moves_made = 0
        skipped_adjacent = 0
        pos = {}

        for y in range(board.height):
            for x in range(board.width):
                cell = board.get(x, y)
                if isinstance(cell, dict) and "content" in cell:
                    pos[cell["content"]] = (x, y)

        total_words = len(pos)

        for current_word in list(pos.keys()):
            if current_word not in pos:
                continue

            curr_x, curr_y = pos[current_word]

            associated_groups = [
                gid
                for gid, members in word_groups.items()
                if current_word in members
            ]

            group_members = set()
            for gid in associated_groups:
                group_members.update(word_groups[gid])

            if len(group_members) <= 1:
                continue

            followers = next_words.get(current_word, set())
            follower_is_adjacent = False
            for follower in followers:
                if follower in pos:
                    fx, fy = pos[follower]
                    if abs(fx - curr_x) <= 1 and abs(fy - curr_y) <= 1:
                        follower_is_adjacent = True
                        break

            if follower_is_adjacent:
                skipped_adjacent += 1
                continue

            other_coords = [
                pos[m] for m in group_members if m in pos and m != current_word
            ]
            if not other_coords:
                continue

            centroid_x = sum(c[0] for c in other_coords) / len(other_coords)
            centroid_y = sum(c[1] for c in other_coords) / len(other_coords)

            dx = (
                1 if centroid_x > curr_x else (-1 if centroid_x < curr_x else 0)
            )
            dy = (
                1 if centroid_y > curr_y else (-1 if centroid_y < curr_y else 0)
            )

            new_x, new_y = curr_x + dx, curr_y + dy

            if board._is_valid(new_x, new_y) and (
                new_x != curr_x or new_y != curr_y
            ):
                target_cell = board.get(new_x, new_y)
                if target_cell == "." or target_cell is None:
                    curr_cell_data = board.get(curr_x, curr_y)
                    board.set(new_x, new_y, curr_cell_data)
                    board.set(curr_x, curr_y, ".")
                    pos[current_word] = (new_x, new_y)
                    moves_made += 1

        progress = (epoch / epochs) * 100
        print(
            f"Epoch {epoch:2d}/{epochs} [{progress:5.1f}%] | "
            f"Moves: {moves_made:4d} | "
            f"Adjacent Anchors: {skipped_adjacent:4d}/{total_words}"
        )

    print("--- Training Complete ---\n")


def save_model(filename):
    """Saves all relevant state variables to disk."""
    state = {
        "board": board,
        "word_groups": word_groups,
        "sentences": sentences,
        "words": words,
        "semantic_words": semantic_words,
    }
    with open(filename, "wb") as f:
        pickle.dump(state, f)
    print(f"Trained model saved to '{filename}'.")

if __name__ == "__main__":
    board, word_groups, sentences = setup()
    train(board, word_groups, sentences, words, epochs=25)
    save_model(model_filename)

    board.display()