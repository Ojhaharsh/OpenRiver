import numpy as np
from treys import Card, Evaluator

class RiverHoldemGame:
    def __init__(self, board_cards_str=["Ks", "Th", "7s", "4d", "2s"]):
        # 1. Setup Constants
        self.SUITS = ['s', 'h', 'd', 'c']
        self.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.HANDS = [r+s for r in self.RANKS for s in self.SUITS]
        
        # 2. Setup Evaluator
        self.evaluator = Evaluator()
        self.board = [Card.new(c) for c in board_cards_str]
        
        # --- THE FIX: Use the short codes the GUI expects ---
        # 0="" (Root), 1="b" (Bet), 2="c" (Check), 3="cb" (Check-Bet)
        self.node_names = {0: "", 1: "b", 2: "c", 3: "cb"}
        
        self.actions = {
            0: ["c", "b"], # Check, Bet (Matches GUI class names .act-c, .act-b)
            1: ["f", "c"], # Fold, Call
            2: ["c", "b"], # Check, Bet
            3: ["f", "c"]  # Fold, Call
        }

    def get_payoff_matrix(self):
        """
        Pre-calculates the 52x52 matrix of who wins.
        Returns: matrix where 1.0 = Row Wins, -1.0 = Row Loses
        """
        print("Calculating Hand Strengths...")
        strengths = np.zeros(len(self.HANDS), dtype=np.int32)
        
        for i, hand_str in enumerate(self.HANDS):
            c1 = Card.new(hand_str)
            
            # --- FIX: Check if card is already on the board ---
            if c1 in self.board:
                # This hand is impossible (e.g. holding As when As is on board)
                # We give it a dummy score of -1 so it effectively never wins.
                strengths[i] = -1
                continue
            # --------------------------------------------------

            # Invert score because treys uses 1=Best, 7462=Worst
            strengths[i] = 7500 - self.evaluator.evaluate(self.board, [c1])
            
        # 1 if Row > Col, -1 if Row < Col
        return np.sign(strengths[:, None] - strengths[None, :]).astype(np.float32)

    def get_pot_size(self, node_idx, action_idx):
        """Returns the pot size for a Showdown at this node"""
        # Logic: Ante=2.0. Bet=+1.0. Call matches bet.
        if node_idx == 1 and action_idx == 1: return 4.0 # Bet(1) + Call(1) + Pot(2)
        if node_idx == 2 and action_idx == 0: return 2.0 # Check + Check + Pot(2)
        if node_idx == 3 and action_idx == 1: return 6.0 # Check + Bet(1) + Raise(2) + Call(1) + Pot(2) -> Simplified
        # (For this demo we stick to the simple pot logic we had earlier)
        return 0.0