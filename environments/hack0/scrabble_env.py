import random
import json
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Any
from tqdm.asyncio import tqdm_asyncio
from pydantic import Field, ValidationError, BaseModel # Import BaseModel directly

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# --- System Prompt for the LLM ---
system_prompt = (
    "You are an expert Scrabble player. Your goal is to play words that maximize your score.\n"
    "You may use extremely long chains of thought to deeply consider the board, your rack, "
    "and potential moves, and deliberate with yourself via systematic reasoning processes "
    "to help come to the best possible solution. "
    "You should enclose your thoughts and internal monologue inside <think> </think> tags.\n\n"
    "After your thoughts, provide your chosen move in a strict JSON format. "
    "The JSON should contain: 'word' (the word to play), 'row' (starting row, 0-14), "
    "'col' (starting column, 0-14), and 'direction' ('across' or 'down').\n"
    "Example response: `{\"word\": \"HELLO\", \"row\": 7, \"col\": 7, \"direction\": \"across\", \"thought\": \"<your reasoning>\"}`. "
    "Your response must be a valid JSON object ONLY. Do NOT include any other text outside the JSON."
)

# --- Data Structures for Scrabble Environment ---

# TypedDict for the data item passed from get_next_item (game state input to LLM)
class ScrabbleRow(TypedDict):
    board_state: List[List[str]]
    player_rack: List[str]
    turn_number: int
    current_score: int # Total score for this player

# Pydantic model for LLM's expected response structure
# Inherit from BaseModel directly to avoid 'TypeError: Cannot subclass typing.Any'
class ScrabbleResponse(BaseModel):
    word: str = Field(..., description="The word to play.")
    row: int = Field(..., description="Starting row for the word.")
    col: int = Field(..., description="Starting column for the word.")
    direction: str = Field(..., description="Direction of the word ('across' or 'down').")
    thought: Optional[str] = Field(None, description="LLM's internal thought process.")

# Scrabble specific environment configuration
class ScrabbleEnvConfig(BaseEnvConfig):
    board_size: int = 15
    rack_size: int = 7
    dictionary_path: str = "./data/scrabble_dictionary.txt"
    letter_values: Dict[str, int] =  {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, '_': 0
        }

# --- ScrabbleEnv Class Definition ---
class ScrabbleEnv(BaseEnv):

    name = "scrabble"

    def __init__(
        self,
        config: ScrabbleEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer: List[float] = [] # Tracks validity/score > 0
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List[Dict[str, Any]] = [] # For logging full rollouts

        # Scrabble game state
        self.dictionary: set[str] = set()
        self.board: List[List[str]] = []
        self.letter_bag: List[str] = []
        self.player_rack: List[str] = []
        self.current_turn: int = 0
        self.total_player_score: int = 0
        self.center_square: Tuple[int, int] = (
            self.config.board_size // 2,
            self.config.board_size // 2,
        )

        # Pre-defined board multipliers (standard Scrabble board)
        self.board_multiplier_map: Dict[Tuple[int, int], Tuple[str, int]] = {
            (0, 0): ('TW', 3), (0, 7): ('TW', 3), (0, 14): ('TW', 3),
            (7, 0): ('TW', 3), (7, 14): ('TW', 3), (14, 0): ('TW', 3),
            (14, 7): ('TW', 3), (14, 14): ('TW', 3),
            (1, 1): ('TW', 3), (2, 2): ('TW', 3), (3, 3): ('TW', 3), (4, 4): ('TW', 3),
            (1, 13): ('TW', 3), (2, 12): ('TW', 3), (3, 11): ('TW', 3), (4, 10): ('TW', 3),
            (13, 1): ('TW', 3), (12, 2): ('TW', 3), (11, 3): ('TW', 3), (10, 4): ('TW', 3),
            (13, 13): ('TW', 3), (12, 12): ('TW', 3), (11, 11): ('TW', 3), (10, 10): ('TW', 3),
            (7, 7): ('DW', 2), # Center square
            (0, 3): ('DL', 2), (0, 11): ('DL', 2), (2, 6): ('DL', 2), (2, 8): ('DL', 2),
            (3, 0): ('DL', 2), (3, 7): ('DL', 2), (3, 14): ('DL', 2), (6, 2): ('DL', 2),
            (6, 6): ('DL', 2), (6, 8): ('DL', 2), (6, 12): ('DL', 2), (7, 3): ('DL', 2),
            (7, 11): ('DL', 2), (8, 2): ('DL', 2), (8, 6): ('DL', 2), (8, 8): ('DL', 2),
            (8, 12): ('DL', 2), (11, 0): ('DL', 2), (11, 7): ('DL', 2), (11, 14): ('DL', 2),
            (12, 6): ('DL', 2), (12, 8): ('DL', 2), (14, 3): ('DL', 2), (14, 11): ('DL', 2),
            (1, 5): ('TL', 3), (1, 9): ('TL', 3), (5, 1): ('TL', 3), (5, 5): ('TL', 3),
            (5, 9): ('TL', 3), (5, 13): ('TL', 3), (9, 1): ('TL', 3), (9, 5): ('TL', 3),
            (9, 9): ('TL', 3), (9, 13): ('TL', 3), (13, 5): ('TL', 3), (13, 9): ('TL', 3),
        }

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = ScrabbleEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=4,
            max_num_workers=2, # Keep low for initial testing
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=10000,
            batch_size=8,
            steps_per_eval=200,
            max_token_length=1024,
            wandb_name="scrabble_llm_rl",
            board_size=15,
            rack_size=7,
            dictionary_path="./data/scrabble_dictionary.txt",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=16, # Fewer requests for faster eval
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/valid_move_rate"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = []

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = []

        wandb_metrics["game/current_turn"] = self.current_turn
        wandb_metrics["game/total_player_score"] = self.total_player_score
        wandb_metrics["game/player_rack_size"] = len(self.player_rack)
        wandb_metrics["game/letters_in_bag"] = len(self.letter_bag)

        if self.rollouts_for_wandb:
            self.add_rollouts_for_wandb(self.rollouts_for_wandb)
            self.rollouts_for_wandb = []

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        print(f"Loading dictionary from: {self.config.dictionary_path}")
        try:
            with open(self.config.dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.dictionary.add(line.strip().upper())
            print(f"Loaded {len(self.dictionary)} words into dictionary.")
        except FileNotFoundError:
            print(f"Dictionary file not found at {self.config.dictionary_path}. Please create it.")
            raise

        self._initialize_game()
        print("Scrabble environment setup complete.")

    def save_checkpoint(self, step: int, data: Optional[Dict] = None):
        if data is None:
            data = {}
        data["current_turn"] = self.current_turn
        data["total_player_score"] = self.total_player_score
        data["board"] = self.board
        data["player_rack"] = self.player_rack
        data["letter_bag"] = self.letter_bag
        super().save_checkpoint(step, data)
        print(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, data: Dict):
        self.current_turn = data.get("current_turn", 0)
        self.total_player_score = data.get("total_player_score", 0)
        self.board = data.get("board", [[' ' for _ in range(self.config.board_size)] for _ in range(self.config.board_size)])
        self.player_rack = data.get("player_rack", [])
        self.letter_bag = data.get("letter_bag", self._initialize_letter_bag())
        print(f"Checkpoint loaded. Resuming from turn {self.current_turn}, score {self.total_player_score}")

    # --- Scrabble Game Logic Helpers ---
    def _initialize_letter_bag(self) -> List[str]:
        letters = []
        letter_counts = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9,
            'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6,
            'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, '_': 2
        }
        for letter, count in letter_counts.items():
            letters.extend([letter] * count)
        random.shuffle(letters)
        print(f"Initialized letter bag with {len(letters)} tiles.")
        return letters

    def _initialize_game(self):
        self.board = [[' ' for _ in range(self.config.board_size)] for _ in range(self.config.board_size)]
        self.letter_bag = self._initialize_letter_bag()
        self.player_rack = self._draw_letters(self.config.rack_size)
        self.current_turn = 1
        self.total_player_score = 0
        print("New Scrabble game initialized.")
        print(f"Initial Rack: {self.player_rack}")

    def _draw_letters(self, num_letters: int) -> List[str]:
        drawn = []
        for _ in range(num_letters):
            if not self.letter_bag:
                self.log.warning("Letter bag is empty. Cannot draw more letters.")
                break
            drawn.append(self.letter_bag.pop())
        return drawn

    def _get_used_letters(self, word: str, row: int, col: int, direction: str,
                          rack: List[str], original_board: List[List[str]]) -> List[str]:
        used = []
        rack_copy = list(rack)
        word_idx = 0

        while word_idx < len(word):
            char_to_place = word[word_idx]
            r, c = (row + word_idx, col) if direction == 'down' else (row, col + word_idx)

            if original_board[r][c] == ' ':
                if char_to_place in rack_copy:
                    used.append(char_to_place)
                    rack_copy.remove(char_to_place)
                elif '_' in rack_copy:
                    used.append('_')
                    rack_copy.remove('_')
            word_idx += 1
        return used

    def _validate_and_score_move(self, board_snapshot: List[List[str]], rack: List[str],
                                 word: str, row: int, col: int, direction: str) -> Tuple[int, bool, List[List[str]], str]:
        temp_board = [r[:] for r in board_snapshot]
        is_valid = True
        score = 0
        reason_invalid = ""
        placed_cells = []
        used_rack_letters_for_scoring = []

        word_upper = word.upper()

        if not (0 <= row < self.config.board_size and 0 <= col < self.config.board_size):
            reason_invalid = "Start position out of bounds."
            return 0, False, board_snapshot, reason_invalid

        if word_upper not in self.dictionary:
            reason_invalid = f"Word '{word_upper}' not in dictionary."
            return 0, False, board_snapshot, reason_invalid

        word_length = len(word_upper)
        if direction == 'across':
            if col + word_length > self.config.board_size:
                reason_invalid = "Word goes off board (across)."
                return 0, False, board_snapshot, reason_invalid
        elif direction == 'down':
            if row + word_length > self.config.board_size:
                reason_invalid = "Word goes off board (down)."
                return 0, False, board_snapshot, reason_invalid
        else:
            reason_invalid = "Invalid direction. Must be 'across' or 'down'."
            return 0, False, board_snapshot, reason_invalid

        current_rack = list(rack)
        num_new_tiles_placed = 0
        word_multiplier = 1

        for i, char_of_word in enumerate(word_upper):
            r, c = (row + i, col) if direction == 'down' else (row, col + i)
            cell_char = temp_board[r][c]

            if cell_char == ' ':
                num_new_tiles_placed += 1
                if char_of_word in current_rack:
                    current_rack.remove(char_of_word)
                    used_rack_letters_for_scoring.append(char_of_word)
                elif '_' in current_rack:
                    current_rack.remove('_')
                    used_rack_letters_for_scoring.append('_')
                else:
                    reason_invalid = f"Not enough letters on rack for '{char_of_word}' at ({r},{c})."
                    return 0, False, board_snapshot, reason_invalid
                temp_board[r][c] = char_of_word
                placed_cells.append((r, c, ' '))

                if (r, c) in self.board_multiplier_map:
                    m_type, m_val = self.board_multiplier_map[(r, c)]
                    if m_type == 'DL':
                        score += self.config.letter_values.get(char_of_word, 0) * (m_val - 1)
                    elif m_type == 'TL':
                        score += self.config.letter_values.get(char_of_word, 0) * (m_val - 1) * 2
                    elif m_type == 'DW' or m_type == 'TW':
                        word_multiplier *= m_val

            elif cell_char == char_of_word:
                placed_cells.append((r, c, cell_char))
            else:
                reason_invalid = f"Conflict at ({r},{c}): Board has '{cell_char}', word needs '{char_of_word}'."
                return 0, False, board_snapshot, reason_invalid

        if num_new_tiles_placed == 0:
            reason_invalid = "No new tiles placed on the board."
            return 0, False, board_snapshot, reason_invalid

        if self.current_turn == 1:
            center_covered = False
            for r, c, _ in placed_cells:
                if (r, c) == self.center_square:
                    center_covered = True
                    break
            if not center_covered:
                reason_invalid = "First word must cover the center square (7,7)."
                return 0, False, board_snapshot, reason_invalid
        else:
            connects_to_existing = False
            for r, c, original_char in placed_cells:
                if original_char != ' ':
                    connects_to_existing = True
                    break
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                for nr, nc in neighbors:
                    if 0 <= nr < self.config.board_size and 0 <= nc < self.config.board_size:
                        if board_snapshot[nr][nc] != ' ':
                            connects_to_existing = True
                            break
                if connects_to_existing:
                    break
            if not connects_to_existing:
                reason_invalid = "Word must connect to an existing word on the board."
                return 0, False, board_snapshot, reason_invalid

        current_word_score = 0
        for i, char_of_word in enumerate(word_upper):
            r, c = (row + i, col) if direction == 'down' else (row, col + i)
            letter_value = self.config.letter_values.get(char_of_word, 0)
            if (r, c, ' ') in placed_cells: # It's a new tile
                current_word_score += letter_value
            else: # It's an existing tile, just add base value
                current_word_score += letter_value

        perpendicular_word_score = 0
        for r, c, original_char in placed_cells:
            if original_char != ' ':
                continue

            perpendicular_word_dir = 'across' if direction == 'down' else 'down'
            perpendicular_word = ''
            perp_start_r, perp_start_c = r, c

            if perpendicular_word_dir == 'across':
                temp_c = c
                while temp_c >= 0 and temp_board[r][temp_c] != ' ':
                    perp_start_c = temp_c
                    temp_c -= 1
            else:
                temp_r = r
                while temp_r >= 0 and temp_board[temp_r][c] != ' ':
                    perp_start_r = temp_r
                    temp_r -= 1

            if perpendicular_word_dir == 'across':
                temp_c = perp_start_c
                while temp_c < self.config.board_size and temp_board[r][temp_c] != ' ':
                    perpendicular_word += temp_board[r][temp_c]
                    temp_c += 1
            else:
                temp_r = perp_start_r
                while temp_r < self.config.board_size and temp_board[temp_r][c] != ' ':
                    perpendicular_word += temp_board[temp_r][c]
                    temp_r += 1

            if len(perpendicular_word) > 1:
                if perpendicular_word not in self.dictionary:
                    reason_invalid = f"Invalid perpendicular word: '{perpendicular_word}'."
                    return 0, False, board_snapshot, reason_invalid
                perp_score = 0
                for p_char in perpendicular_word:
                    perp_score += self.config.letter_values.get(p_char, 0)
                perpendicular_word_score += perp_score

        total_move_score = (current_word_score + perpendicular_word_score) * word_multiplier

        if len(used_rack_letters_for_scoring) == self.config.rack_size:
            total_move_score += 50

        print(f"Move '{word_upper}' at ({row},{col}) {direction}. Base: {current_word_score}, Perpendicular: {perpendicular_word_score}, Word Multiplier: {word_multiplier}, Bingo: {50 if len(used_rack_letters_for_scoring) == self.config.rack_size else 0}. Total: {total_move_score}")

        return total_move_score, is_valid, temp_board, reason_invalid

    async def rollout_and_score_eval(self, board_state: List[List[str]], player_rack: List[str]) -> float:
        board_str = "\n".join([" ".join(row) for row in board_state])
        letters = ', '.join(player_rack)
        prompt = (
            "You are playing Scrabble. Here is the current board:\n"
            f"```\n{board_str}\n```\n"
            f"Your letters: {letters}\n"
            "What word will you play? Provide the word, its starting row (0-14), column (0-14), "
            "and direction ('across' or 'down'). Respond in JSON format only.\n"
            "Your response must be a valid JSON object like: "
            "`{\"word\": \"HELLO\", \"row\": 7, \"col\": 7, \"direction\": \"across\"}`."
        )

        try:
            completion = await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
                json_mode=True,
                split="eval",
            )
            response_content = completion.choices[0].message.content
            move_data = ScrabbleResponse.model_validate_json(response_content)

            score, is_valid, _, reason = self._validate_and_score_move(
                board_state, player_rack, move_data.word, move_data.row, move_data.col, move_data.direction
            )
            print(f"Eval move: {move_data.word}, Valid: {is_valid}, Score: {score}, Reason: {reason}")
            return float(score if is_valid else 0.0)
        except (ValidationError, json.JSONDecodeError, IndexError) as e:
            print(f"Eval LLM response parsing error: {e}. Raw response: {completion.choices[0].message.content if completion.choices else 'N/A'}")
            return 0.0

    async def evaluate(self, *args, **kwargs):
        num_eval_games = 5 # Reduced for quicker testing
        eval_scores = []

        print(f"Starting evaluation over {num_eval_games} simulated games.")

        eval_tasks = []
        for _ in range(num_eval_games):
            temp_board = [[' ' for _ in range(self.config.board_size)] for _ in range(self.config.board_size)]
            temp_rack = self._draw_letters(self.config.rack_size)
            eval_tasks.append(self.rollout_and_score_eval(temp_board, temp_rack))

        scores = await tqdm_asyncio.gather(*eval_tasks)
        avg_score = sum(scores) / len(scores) if scores else 0
        valid_moves_count = sum(1 for s in scores if s > 0)
        valid_rate = valid_moves_count / len(scores) if scores else 0

        self.eval_metrics.append(("eval/avg_move_score", avg_score))
        self.eval_metrics.append(("eval/valid_move_rate", valid_rate))
        print(f"Evaluation complete. Avg Score: {avg_score:.2f}, Valid Move Rate: {valid_rate:.2f}")

    async def collect_trajectories(
        self, item: ScrabbleRow
    ) -> Tuple[ScoredDataGroup, list[Any]]: # Changed list[Item] to list[Any] to match type_definitions
        board_str = "\n".join([" ".join(row) for row in item["board_state"]])
        letters = ', '.join(item["player_rack"])
        prompt = (
            "You are playing Scrabble. Here is the current board:\n"
            f"```\n{board_str}\n```\n"
            f"Your letters: {letters}\n"
            "What word will you play? Provide the word, its starting row (0-14), column (0-14), "
            "and direction ('across' or 'down'). Respond in JSON format only.\n"
            "Your response must be a valid JSON object like: "
            "`{\"word\": \"HELLO\", \"row\": 7, \"col\": 7, \"direction\": \"across\", \"thought\": \"<your reasoning>\"}`."
        )

        user_message = {"role": "user", "content": prompt}

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            json_mode=True,
            temperature=0.7
        )

        to_score_items = []
        for chat_completion in chat_completions.choices:
            full_messages = [
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            ]
            to_score_items.append(
                {
                    "messages": full_messages,
                    "board_state": item["board_state"],
                    "player_rack": item["player_rack"],
                    "finish_reason": chat_completion.finish_reason,
                    "llm_raw_response": chat_completion.message.content
                }
            )

        scored_data_group = await self.score(to_score_items)

        return scored_data_group, []

    async def score(
        self, rollout_group_data: List[Dict[str, Any]]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["prompt"] = rollout_group_data[0]["messages"][1]["content"]
        scores["response"] = []

        group_rewards = []
        best_score_in_group = -float('inf')
        best_move_applied = False
        board_snapshot_for_group = rollout_group_data[0]["board_state"]
        rack_snapshot_for_group = rollout_group_data[0]["player_rack"]

        temp_board_after_best_move = [r[:] for r in board_snapshot_for_group]
        best_used_letters_for_scoring = []

        shuffled_rollout_group_data = list(rollout_group_data)
        random.shuffle(shuffled_rollout_group_data)

        for item in shuffled_rollout_group_data:
            llm_response_text = item["messages"][-1]["content"]
            valid_move_score = 0.0
            is_valid_move = False
            move_reason = "Malformed response or other error."

            try:
                move_data = ScrabbleResponse.model_validate_json(llm_response_text)
                word = move_data.word
                row, col = move_data.row, move_data.col
                direction = move_data.direction

                move_score, is_valid_move, new_board_state, move_reason = self._validate_and_score_move(
                    board_snapshot_for_group, rack_snapshot_for_group, word, row, col, direction
                )

                if is_valid_move:
                    valid_move_score = float(move_score)
                    print(f"Valid LLM move: {word} ({row},{col}) {direction}, Score: {move_score}")

                    if move_score > best_score_in_group:
                        best_score_in_group = move_score
                        temp_board_after_best_move = new_board_state
                        best_used_letters_for_scoring = self._get_used_letters(
                            word, row, col, direction, rack_snapshot_for_group, board_snapshot_for_group
                        )
                        best_move_applied = True
                else:
                    print(f"Invalid LLM move: {word} ({row},{col}) {direction}, Reason: {move_reason}")

            except (ValidationError, json.JSONDecodeError) as e:
                self.log.warning(f"Failed to parse LLM JSON for scoring: {e}. Raw: {llm_response_text[:100]}...")
                move_reason = f"JSON parsing error: {e}"
            except Exception as e:
                print(f"Unexpected error during move processing: {e}. Raw: {llm_response_text[:100]}...")
                move_reason = f"Internal error: {e}"

            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            if len([1 for i in masks if i != -100]) < 10:
                print("Skipping very short tokenized response.")
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(valid_move_score if is_valid_move else -1.0)
            scores["response"].append(llm_response_text)

            self.rollouts_for_wandb.append({
                "prompt": item["messages"][1]["content"],
                "response": llm_response_text,
                "score": valid_move_score,
                "valid": is_valid_move,
                "reason": move_reason
            })

            group_rewards.append(1.0 if is_valid_move and valid_move_score > 0 else 0.0)

        if best_move_applied:
            self.board = temp_board_after_best_move
            self.total_player_score += best_score_in_group
            self.current_turn += 1

            temp_rack_for_update = list(self.player_rack)
            for used_char in best_used_letters_for_scoring:
                if used_char in temp_rack_for_update:
                    temp_rack_for_update.remove(used_char)
                else:
                    self.log.warning(f"'_get_used_letters' reported using '{used_char}' but not found on rack for removal. Logic error?")
            self.player_rack = temp_rack_for_update
            self.player_rack.extend(self._draw_letters(self.config.rack_size - len(self.player_rack)))
            print(f"Turn {self.current_turn-1} complete. Best score: {best_score_in_group}. Total score: {self.total_player_score}.")
            print(f"New Rack: {self.player_rack}")
        else:
            self.current_turn += 1
            self.log.warning(f"Turn {self.current_turn-1} complete. No valid move found in group. Total score: {self.total_player_score}.")
            self.player_rack = self._draw_letters(self.config.rack_size)
            print(f"Player exchanged rack due to no valid moves. New Rack: {self.player_rack}")

        if all(s > 0 for s in scores["scores"]):
            token_lengths = [len(s_tokens) for s_tokens in scores["tokens"]]
            if max(token_lengths) == 0:
                return None

            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5

            for idx, length in enumerate(token_lengths):
                if length <= length_threshold:
                    scores["scores"][idx] *= 1.0
                else:
                    percentage_of_range = (length - length_threshold) / (max_allowed_length - length_threshold)
                    percentage_of_range = min(percentage_of_range, 1.0)
                    scores["scores"][idx] *= (1.0 - percentage_of_range)

        if len(scores["scores"]) > 1 and all(s == scores["scores"][0] for s in scores["scores"]):
            print("All scores in group are identical. Returning None.")
            return None

        if group_rewards:
            self.percent_correct_buffer.append(sum(group_rewards) / len(group_rewards))

        return scores

    async def get_next_item(self) -> ScrabbleRow:
        if not self.player_rack and not self.letter_bag:
            print("Game over: No more letters to draw or play. Resetting game.")
            self._initialize_game()

        return ScrabbleRow(
            board_state=[row[:] for row in self.board],
            player_rack=self.player_rack[:],
            turn_number=self.current_turn,
            current_score=self.total_player_score,
        )

if __name__ == "__main__":
    ScrabbleEnv.cli()