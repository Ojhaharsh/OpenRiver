import numpy as np
import time
from typing import TYPE_CHECKING
from src.utils.logging_config import get_logger
from src.utils.strategy_cache import StrategyCache

if TYPE_CHECKING:
    from src.games.river_holdem import RiverHoldemGame

logger = get_logger(__name__)


class VectorCFR:
    """
    Vectorized Counterfactual Regret Minimization algorithm.
    
    Uses flat NumPy arrays instead of tree traversal for fast poker solving.
    Converges to Nash equilibrium through regret minimization.
    """
    
    def __init__(self, game: "RiverHoldemGame") -> None:
        """
        Initialize the CFR solver with a game instance.
        
        Args:
            game: RiverHoldemGame instance defining the game structure
        """
        self.game = game
        self.num_hands = len(game.HANDS)
        self.num_nodes = len(game.node_names)
        
        # Memory Allocation (Flat Arrays)
        self.regret_sum = np.zeros((self.num_nodes, self.num_hands, 2), dtype=np.float32)
        self.strategy_sum = np.zeros((self.num_nodes, self.num_hands, 2), dtype=np.float32)
        self.payoff_matrix = game.get_payoff_matrix()
    
    def save_strategy(self, board_cards: list[str]) -> bool:
        """
        Save current strategy to cache.
        
        Args:
            board_cards: List of board cards
            
        Returns:
            True if successful
        """
        solver_data = {
            "strategy_sum": self.strategy_sum,
            "regret_sum": self.regret_sum,
            "node_names": self.game.node_names,
            "actions": self.game.actions,
            "hands": self.game.HANDS,
        }
        return StrategyCache.save_strategy(board_cards, solver_data)
    
    def load_strategy(self, board_cards: list[str]) -> bool:
        """
        Load strategy from cache if available.
        
        Args:
            board_cards: List of board cards
            
        Returns:
            True if strategy was loaded, False otherwise
        """
        data = StrategyCache.load_strategy(board_cards)
        if data is None:
            return False
        
        try:
            if "strategy_sum" in data:
                self.strategy_sum = data["strategy_sum"]
            if "regret_sum" in data:
                self.regret_sum = data["regret_sum"]
            logger.info(f"Strategy loaded for board: {' '.join(board_cards)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load strategy data: {str(e)}")
            return False

    def get_strategy(self, node_idx: int, hand_idx: int, reach_prob: float) -> np.ndarray:
        """
        Compute strategy for a hand at a specific node using regret matching.
        
        Args:
            node_idx: Game tree node index
            hand_idx: Index of the hand (0-51)
            reach_prob: Probability of reaching this node
            
        Returns:
            Strategy vector [0, 1] representing action probabilities
        """
        regrets = self.regret_sum[node_idx, hand_idx]
        pos_regret = np.maximum(regrets, 0)
        sum_regret = np.sum(pos_regret)
        
        if sum_regret > 0:
            strategy = pos_regret / sum_regret
        else:
            strategy = np.array([0.5, 0.5], dtype=np.float32)
            
        self.strategy_sum[node_idx, hand_idx] += strategy * reach_prob
        return strategy

    def train(self, iterations: int = 100000) -> None:
        """
        Run CFR iterations to solve the game.
        
        Args:
            iterations: Number of training iterations to perform
        """
        logger.info(f"Starting Vector CFR for {iterations} iterations...")
        start = time.time()
        
        for _ in range(iterations):
            h1, h2 = np.random.randint(0, self.num_hands, 2)
            if h1 != h2:
                self._cfr_recursive(0, 1.0, 1.0, h1, h2)
                
        dt = time.time() - start
        iters_per_sec = int(iterations/dt)
        logger.info(f"Solved in {dt:.2f}s ({iters_per_sec} iters/sec)")

    def _cfr_recursive(self, node_idx: int, p0: float, p1: float, h1: int, h2: int) -> float:
        """
        Recursive CFR computation for a single game state.
        
        Args:
            node_idx: Current node in game tree
            p0: Reach probability for player 0
            p1: Reach probability for player 1
            h1: Hand index for player 0
            h2: Hand index for player 1
            
        Returns:
            Utility value from this node forward (from current player's perspective)
        """
        player = 0 if (node_idx == 0 or node_idx == 3) else 1
        my_hand = h1 if player == 0 else h2
        
        strategy = self.get_strategy(node_idx, my_hand, p0 if player == 0 else p1)
        util = np.zeros(2, dtype=np.float32)
        
        # --- GRAPH TRAVERSAL (The same fast logic as before) ---
        if node_idx == 0: # Root
            util[0] = -self._cfr_recursive(2, p0*strategy[0], p1, h1, h2)
            util[1] = -self._cfr_recursive(1, p0*strategy[1], p1, h1, h2)
            
        elif node_idx == 1: # Facing Bet
            util[0] = -1.0 # Fold
            # Call (Showdown Pot 2.0)
            win_val = self.payoff_matrix[h1, h2] if player==1 else self.payoff_matrix[h2, h1]
            util[1] = win_val * 2.0
            
        elif node_idx == 2: # Checked To
            # Check (Showdown Pot 1.0)
            win_val = self.payoff_matrix[h1, h2] if player==1 else self.payoff_matrix[h2, h1]
            util[0] = win_val * 1.0
            util[1] = -self._cfr_recursive(3, p0, p1*strategy[1], h1, h2)
            
        elif node_idx == 3: # Check Raise
            util[0] = -1.0 # Fold
            # Call (Showdown Pot 3.0 simplified)
            win_val = self.payoff_matrix[h1, h2] if player==1 else self.payoff_matrix[h2, h1]
            util[1] = win_val * 3.0

        # Regret Update
        node_util = strategy[0]*util[0] + strategy[1]*util[1]
        opp_prob = p1 if player == 0 else p0
        self.regret_sum[node_idx, my_hand] += (util - node_util) * opp_prob
        return node_util