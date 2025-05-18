# environments/scrabble_env.py (initial snippet)
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, Item, ScoredDataGroup
from pydantic import Field
from typing import List, Tuple, Dict, Any, Optional
import asyncio
import random # For simulating game state

# Define your Scrabble-specific Item
class ScrabbleItem(Item):
    game_state: Dict[str, Any] = Field(..., description="Current state of the Scrabble game.")
    turn_number: int = Field(..., description="Current turn number.")
    player_rack: List[str] = Field(..., description="Letters on the current player's rack.")
    board_state: List[List[str]] = Field(..., description="Current state of the Scrabble board.")
    # Add more game state elements as needed (e.g., scores, letter bag)

# Define the structure of the LLM's expected response
class ScrabbleResponse(Item):
    word: str = Field(..., description="The word to play.")
    row: int = Field(..., description="Starting row for the word.")
    col: int = Field(..., description="Starting column for the word.")
    direction: str = Field(..., description="Direction of the word ('across' or 'down').")
    # Add an optional 'thought' field if you want the LLM to explain its reasoning
    thought: Optional[str] = Field(None, description="LLM's internal thought process.")

class ScrabbleEnvConfig(BaseEnvConfig):
    # Add Scrabble-specific configuration here if needed
    board_size: int = 15
    rack_size: int = 7
    dictionary_path: str = "./data/scrabble_dictionary.txt" # You'll need a dictionary file

# environments/scrabble_env.py (continued)
from atroposlib.prompts.base import Request, Generation

class ScrabbleEnv(BaseEnv):
    env_config_cls = ScrabbleEnvConfig # Register your custom config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dictionary = set()
        self.current_game_state: Dict[str, Any] = {} # Will hold the actual game state
        self.current_turn = 0
        self.board = [[' ' for _ in range(self.config.board_size)] for _ in range(self.config.board_size)]
        self.letter_bag = self._initialize_letter_bag() # Implement this
        self.player_rack = []

    async def setup(self):
        # Load dictionary
        with open(self.config.dictionary_path, 'r') as f:
            for line in f:
                self.dictionary.add(line.strip().upper())
        self._initialize_game() # Set up initial board, racks etc.

    def _initialize_letter_bag(self) -> List[str]:
        # Basic Scrabble letter distribution (customize as needed)
        letters = []
        letter_counts = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9,
            'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6,
            'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, '_': 2 # Blank tiles
        }
        for letter, count in letter_counts.items():
            letters.extend([letter] * count)
        random.shuffle(letters)
        return letters

    def _initialize_game(self):
        # For simplicity, start with an empty board and draw initial letters
        self.board = [[' ' for _ in range(self.config.board_size)] for _ in range(self.config.board_size)]
        self.player_rack = self._draw_letters(self.config.rack_size)
        self.current_turn = 1
        self.current_game_state = {
            "board": [row[:] for row in self.board], # Deep copy
            "rack": self.player_rack[:], # Deep copy
            "score": 0 # Simple single player score
        }

    def _draw_letters(self, num_letters: int) -> List[str]:
        drawn = []
        for _ in range(num_letters):
            if not self.letter_bag:
                break # Bag is empty
            drawn.append(self.letter_bag.pop())
        return drawn

    async def get_next_item(self) -> ScrabbleItem:

        # Check if the game has ended (e.g., no more letters, no more moves)
        if not self.player_rack and not self.letter_bag:
            self.log.info("Game ended: No more letters to draw or play.")
            # Optionally, reset the game or signal end to the trainer
            self._initialize_game() # Reset for continuous training
            # Or return None if you want to stop collecting for this worker
            # return None

        return ScrabbleItem(
            game_state=self.current_game_state,
            turn_number=self.current_turn,
            player_rack=self.player_rack,
            board_state=self.board
        )

    async def collect_trajectory(self, item: ScrabbleItem) -> Tuple[Optional[ScoredDataGroup], List[Item]]:

        # Construct a simple board string for the prompt
        board_str = "\n".join([" ".join(row) for row in item.board_state])
        prompt = (
            "You are playing Scrabble. Here is the current board:\n"
            f"```\n{board_str}\n```\n"
            f"Your letters: {', '.join(item.player_rack)}\n"
            "What word will you play? Provide the word, its starting row (0-14), column (0-14), and direction ('across' or 'down').\n"
            "Respond in JSON format like this: `{\"word\": \"HELLO\", \"row\": 7, \"col\": 7, \"direction\": \"across\", \"thought\": \"<your reasoning>\"}`. "
            "Your response must be a valid JSON object only."
        )

        llm_request = Request(
            prompt=prompt,
            # Adjust max_tokens based on expected response length + thought
            max_tokens=self.config.max_token_length,
            temperature=0.7,
            top_p=0.9,
            n=self.config.group_size, # Ask for group_size completions
            stop=["}"], # Stop generation after the JSON object is closed
            json_mode=True # Enable JSON mode if your inference server supports it (like vLLM)
        )

        responses: List[Generation] = await self.request_llm(llm_request)

        scored_responses: List[Dict[str, Any]] = []
        best_score = -float('inf')
        best_move_data = None
        current_board_copy = [row[:] for row in self.board] # Board state before any move

        for i, gen in enumerate(responses):
            try:
                # Attempt to parse the JSON response from the LLM
                llm_output_text = gen.text.strip()
                # Ensure the LLM output is a complete JSON object if json_mode isn't strict
                if not llm_output_text.startswith("{"):
                    llm_output_text = "{" + llm_output_text
                if not llm_output_text.endswith("}"):
                    llm_output_text = llm_output_text + "}"

                move_data = ScrabbleResponse.model_validate_json(llm_output_text)
                word = move_data.word.upper()
                row, col = move_data.row, move_data.col
                direction = move_data.direction.lower()
                thought = move_data.thought

                # Validate the move and calculate score
                score, is_valid, new_board = self._validate_and_score_move(
                    current_board_copy, item.player_rack, word, row, col, direction
                )

                if is_valid:
                    self.log.info(f"LLM proposed valid move: {word} at ({row}, {col}) {direction} - Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_move_data = (word, row, col, direction, new_board, score, thought)
                else:
                    self.log.info(f"LLM proposed invalid move: {word} at ({row}, {col}) {direction} - Reason: Invalid.")
                    score = 0 # Penalize invalid moves

                scored_responses.append({
                    "prompt": prompt,
                    "response": gen.text,
                    "score": float(score), # Ensure score is float
                    "valid_move": is_valid,
                    "word": word,
                    "row": row,
                    "col": col,
                    "direction": direction,
                    "thought": thought
                })

            except Exception as e:
                self.log.error(f"Error parsing LLM response or validating move: {e}. Response: {gen.text}")
                scored_responses.append({
                    "prompt": prompt,
                    "response": gen.text,
                    "score": 0.0, # Assign 0 score for malformed/unparseable responses
                    "valid_move": False,
                    "error": str(e)
                })

        # Update game state based on the *best* valid move from the group
        if best_move_data:
            word, row, col, direction, new_board, score, thought = best_move_data
            self.board = new_board # Update the actual board state
            self.current_game_state["board"] = [row[:] for row in self.board]
            # Remove used letters from rack and draw new ones
            used_letters = self._get_used_letters(word, row, col, direction, item.player_rack, current_board_copy)
            self.player_rack = [l for l in self.player_rack if l not in used_letters] # Naive removal
            self.player_rack.extend(self._draw_letters(self.config.rack_size - len(self.player_rack)))
            self.current_game_state["rack"] = self.player_rack[:]
            self.current_game_state["score"] += score # Add score to total
            self.current_turn += 1
            self.log.info(f"Applying best move: {word}, Score: {score}, Total Score: {self.current_game_state['score']}")
        else:
            self.log.warning("No valid move found in the current group of LLM responses. Player loses turn.")
            # Optionally, penalize or make the LLM draw new tiles
            self.current_turn += 1
            # Still need to draw new letters to prevent stalemate if rack is empty
            if not self.player_rack:
                self.player_rack.extend(self._draw_letters(self.config.rack_size))
                self.current_game_state["rack"] = self.player_rack[:]


        data_group = ScoredDataGroup(
            prompts=[sr["prompt"] for sr in scored_responses],
            responses=[sr["response"] for sr in scored_responses],
            scores=[sr["score"] for sr in scored_responses],
            group_id=self.current_turn, # Use turn number as group_id
            metadata=scored_responses # Store full details for logging/debugging
        )
        # No new backlog items in this simple game simulation
        return data_group, []

    def _validate_and_score_move(self, board: List[List[str]], rack: List[str], word: str, row: int, col: int, direction: str) -> Tuple[int, bool, List[List[str]]]:
        # Basic validation:
        # 1. Check if word is in dictionary.
        # 2. Check if word fits on board.
        # 3. Check if player has letters (on rack or on board).
        # 4. Check if word connects to existing words (first move needs to hit center).
        # 5. Calculate score based on letter values, board multipliers, and new words formed.

        temp_board = [r[:] for r in board] # Work on a copy
        is_valid = False
        score = 0
        used_rack_letters = []

        # Check dictionary
        if word not in self.dictionary:
            self.log.debug(f"Invalid word: {word} not in dictionary.")
            return 0, False, board

        # Check board boundaries and letter availability
        if direction == 'across':
            if col + len(word) > self.config.board_size:
                self.log.debug("Word goes off board (across).")
                return 0, False, board
            for i, char in enumerate(word):
                board_char = temp_board[row][col + i]
                if board_char == ' ': # Space on board, must use rack letter
                    if char in rack:
                        temp_board[row][col + i] = char
                        used_rack_letters.append(char)
                    elif '_' in rack: # Use a blank tile
                        temp_board[row][col + i] = char # Blank tile represents this char
                        used_rack_letters.append('_')
                    else:
                        self.log.debug(f"Missing letter {char} for across placement (row {row}, col {col+i}).")
                        return 0, False, board
                elif board_char != char: # Letter on board but doesn't match word
                    self.log.debug(f"Board conflict at ({row}, {col+i}): {board_char} vs {char}.")
                    return 0, False, board
        elif direction == 'down':
            if row + len(word) > self.config.board_size:
                self.log.debug("Word goes off board (down).")
                return 0, False, board
            for i, char in enumerate(word):
                board_char = temp_board[row + i][col]
                if board_char == ' ':
                    if char in rack:
                        temp_board[row + i][col] = char
                        used_rack_letters.append(char)
                    elif '_' in rack:
                        temp_board[row + i][col] = char
                        used_rack_letters.append('_')
                    else:
                        self.log.debug(f"Missing letter {char} for down placement (row {row+i}, col {col}).")
                        return 0, False, board
                elif board_char != char:
                    self.log.debug(f"Board conflict at ({row+i}, {col}): {board_char} vs {char}.")
                    return 0, False, board
        else:
            self.log.debug("Invalid direction. Must be 'across' or 'down'.")
            return 0, False, board

        # Check if any new letters were placed (must place at least one)
        if not used_rack_letters:
            self.log.debug("No new letters placed on the board.")
            return 0, False, board

        # Check if the first word hits the center (7,7)
        if self.current_turn == 1:
            if not ((direction == 'across' and row == 7 and col <= 7 < col + len(word)) or
                    (direction == 'down' and col == 7 and row <= 7 < row + len(word))):
                self.log.debug("First word must cover center square (7,7).")
                return 0, False, board
            
        score = len(word) * 10
        if len(used_rack_letters) == self.config.rack_size: # Used all letters
            score += 50 # Bingo bonus

        is_valid = True # If we reached here, assumed valid for this basic check
        return score, is_valid, temp_board

    def _get_used_letters(self, word: str, row: int, col: int, direction: str, rack: List[str], original_board: List[List[str]]) -> List[str]:
        # Determine which letters from the rack were used for the word
        used = []
        rack_copy = rack[:]
        if direction == 'across':
            for i, char in enumerate(word):
                if original_board[row][col + i] == ' ':
                    if char in rack_copy:
                        used.append(char)
                        rack_copy.remove(char)
                    elif '_' in rack_copy:
                        used.append('_') # A blank tile was used for this char
                        rack_copy.remove('_')
        elif direction == 'down':
            for i, char in enumerate(word):
                if original_board[row + i][col] == ' ':
                    if char in rack_copy:
                        used.append(char)
                        rack_copy.remove(char)
                    elif '_' in rack_copy:
                        used.append('_')
                        rack_copy.remove('_')
        return used


    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        wandb_metrics["game/current_score"] = self.current_game_state["score"]
        wandb_metrics["game/turn_number"] = self.current_turn
        await super().wandb_log(wandb_metrics)


    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        return (
            cls.env_config_cls(
                group_size=4, # How many responses the LLM generates per turn for scoring
                max_token_length=512, # Max tokens for LLM generation
                use_wandb=True,
                wandb_project="scrabble_llm_rl",
                board_size=15,
                rack_size=7,
                dictionary_path="./data/scrabble_dictionary.txt" # Ensure this path is correct
            ),
            [APIServerConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")] # Or your vLLM model
        )