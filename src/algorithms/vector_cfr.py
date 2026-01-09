import numpy as np
import time

class VectorCFR:
    def __init__(self, game):
        self.game = game
        self.num_hands = len(game.HANDS)
        self.num_nodes = len(game.node_names)
        
        # Memory Allocation (Flat Arrays)
        self.regret_sum = np.zeros((self.num_nodes, self.num_hands, 2), dtype=np.float32)
        self.strategy_sum = np.zeros((self.num_nodes, self.num_hands, 2), dtype=np.float32)
        self.payoff_matrix = game.get_payoff_matrix()

    def get_strategy(self, node_idx, hand_idx, reach_prob):
        regrets = self.regret_sum[node_idx, hand_idx]
        pos_regret = np.maximum(regrets, 0)
        sum_regret = np.sum(pos_regret)
        
        if sum_regret > 0:
            strategy = pos_regret / sum_regret
        else:
            strategy = np.array([0.5, 0.5], dtype=np.float32)
            
        self.strategy_sum[node_idx, hand_idx] += strategy * reach_prob
        return strategy

    def train(self, iterations=100000):
        print(f"Starting Vector CFR for {iterations} iterations...")
        start = time.time()
        
        for _ in range(iterations):
            h1, h2 = np.random.randint(0, self.num_hands, 2)
            if h1 != h2:
                self._cfr_recursive(0, 1.0, 1.0, h1, h2)
                
        dt = time.time() - start
        print(f"Solved in {dt:.2f}s ({int(iterations/dt)} iters/sec)")

    def _cfr_recursive(self, node_idx, p0, p1, h1, h2):
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