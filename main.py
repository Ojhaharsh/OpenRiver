import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.games.river_holdem import RiverHoldemGame
from src.algorithms.vector_cfr import VectorCFR

# --- GLOBAL STATE ---
# We store the current solver instance here so the API can access it.
current_solver = None
current_game = None
current_board_cards = ["Ks", "Th", "7s", "4d", "2s"] # Default storage


def build_solver(board_str):
    """Helper to create a new solver from scratch with a new board"""
    global current_solver, current_game, current_board_cards
    
    # Update global board state
    current_board_cards = board_str.split()
    
    # Initialize Game & Solver
    current_game = RiverHoldemGame(current_board_cards)
    current_solver = VectorCFR(current_game)

# Initialize with a default board so the server doesn't crash on load
build_solver("Ks Th 7s 4d 2s")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/train")
def train_endpoint(board: str = "Ks Th 7s 4d 2s", iters: int = 50000):
    """
    1. Receives new board (e.g. 'Ah Kd 2s 9c 5h')
    2. Rebuilds the math model.
    3. Runs the solver.
    """
    print(f"Received Request: Solve board '{board}' for {iters} iterations.")
    
    # Rebuild the engine for the new board
    build_solver(board)
    
    # Train
    current_solver.train(iters)
    return {"status": "solved", "board": board}

@app.get("/solution")
def solution_endpoint():
    # START OF CHANGE
    # We add a top-level "board" key so the GUI knows what cards to draw
    res = {"board": current_board_cards} 
    # END OF CHANGE
    
    for idx, name in current_game.node_names.items():
        res[name] = {"actions": current_game.actions[idx], "hands": {}}
        for h_idx, hand_str in enumerate(current_game.HANDS):
            s_sum = current_solver.strategy_sum[idx, h_idx]
            norm = np.sum(s_sum)
            strat = (s_sum / norm).tolist() if norm > 0 else [0.5, 0.5]
            res[name]["hands"][hand_str] = strat
    return res


if __name__ == "__main__":
    # Pre-train the default board just to verify logic works
    print("Pre-checking default board...")
    current_solver.train(1000) 
    print("Ready! Starting Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)