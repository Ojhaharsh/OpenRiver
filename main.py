import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.games.river_holdem import RiverHoldemGame
from src.algorithms.vector_cfr import VectorCFR
from src.utils.validation import validate_board_cards, validate_iterations
from src.exceptions import InvalidBoardError, InvalidIterationsError, SolverNotInitializedError
from src.utils.logging_config import setup_logging, get_logger
from src.config import get_config
from src.utils.strategy_cache import StrategyCache

# Initialize configuration and logging
config = get_config()
logger = setup_logging(log_level=config.get("log_level", "INFO"))
logger.info("Poker Solver startup")

# --- GLOBAL STATE ---
# We store the current solver instance here so the API can access it.
current_solver = None
current_game = None
current_board_cards = ["Ks", "Th", "7s", "4d", "2s"] # Default storage


def build_solver(board_input):
    """Helper to create a new solver from scratch with a new board"""
    global current_solver, current_game, current_board_cards
    
    # If board_input is already parsed (list), use it; otherwise parse it
    if isinstance(board_input, list):
        current_board_cards = board_input
    else:
        is_valid, error_msg, parsed_cards = validate_board_cards(board_input)
        if not is_valid:
            raise ValueError(f"Invalid board: {error_msg}")
        current_board_cards = parsed_cards
    
    logger.info(f"Rebuilding solver for board: {' '.join(current_board_cards)}")
    
    # Initialize Game & Solver
    current_game = RiverHoldemGame(current_board_cards)
    current_solver = VectorCFR(current_game)

# Initialize with a default board so the server doesn't crash on load
try:
    default_board = config.get("default_board")
    board_str = " ".join(default_board)
    build_solver(board_str)
    logger.info("Default board initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize default board: {str(e)}")

app = FastAPI()

# Configure CORS with settings from config
cors_origins = config.get("cors_origins", ["*"])
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])
logger.info(f"CORS configured for origins: {cors_origins}")

@app.get("/train")
def train_endpoint(board: str = None, iters: int = None):
    """
    1. Receives new board (e.g. 'Ah Kd 2s 9c 5h')
    2. Validates input
    3. Checks cache for existing solution
    4. Rebuilds the math model if needed
    5. Runs the solver.
    """
    # Use config defaults if not provided
    if board is None:
        board_list = config.get("default_board")
        board = " ".join(board_list)
    if iters is None:
        iters = config.get("default_iterations", 50000)
    
    try:
        # Validate board input
        is_valid, error_msg, parsed_cards = validate_board_cards(board)
        if not is_valid:
            raise InvalidBoardError(error_msg)
        
        # Validate iterations
        is_valid, error_msg = validate_iterations(iters)
        if not is_valid:
            raise InvalidIterationsError(error_msg)
        
        logger.info(f"Training request: board='{board}', iters={iters}")
        
        # Rebuild the engine for the new board
        build_solver(parsed_cards)
        
        # Check if strategy exists in cache
        if current_solver.load_strategy(parsed_cards):
            logger.info(f"Using cached strategy for board: {board}")
            return {"status": "solved", "board": board, "cached": True}
        
        # Train if not cached
        current_solver.train(iters)
        
        # Save strategy to cache
        current_solver.save_strategy(parsed_cards)
        
        logger.info(f"Training completed for board: {board}")
        return {"status": "solved", "board": board, "cached": False}
    
    except InvalidBoardError as e:
        logger.warning(f"Invalid board input: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid board: {str(e)}")
    except InvalidIterationsError as e:
        logger.warning(f"Invalid iterations input: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid iterations: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during training")

@app.get("/solution")
def solution_endpoint():
    """Return the current solution with board cards"""
    try:
        if current_solver is None or current_game is None:
            raise SolverNotInitializedError("Solver not initialized. Call /train first.")
        
        logger.debug("Retrieving current solution")
        res = {"board": current_board_cards} 
        
        for idx, name in current_game.node_names.items():
            res[name] = {"actions": current_game.actions[idx], "hands": {}}
            for h_idx, hand_str in enumerate(current_game.HANDS):
                s_sum = current_solver.strategy_sum[idx, h_idx]
                norm = np.sum(s_sum)
                strat = (s_sum / norm).tolist() if norm > 0 else [0.5, 0.5]
                res[name]["hands"][hand_str] = strat
        return res
    
    except SolverNotInitializedError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error retrieving solution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving solution")


@app.get("/cache/list")
def list_cache_endpoint():
    """List all cached board strategies"""
    try:
        cached_boards = StrategyCache.list_cached_boards()
        logger.info(f"Listed {len(cached_boards)} cached strategies")
        return {"cached_boards": [" ".join(board) for board in cached_boards]}
    except Exception as e:
        logger.error(f"Error listing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing cache")


@app.delete("/cache/clear")
def clear_cache_endpoint():
    """Clear all cached strategies"""
    try:
        StrategyCache.clear_cache()
        logger.info("Cache cleared via API")
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing cache")


if __name__ == "__main__":
    # Pre-train the default board just to verify logic works
    warmup_iters = config.get("warmup_iterations", 1000)
    logger.info(f"Pre-checking default board with {warmup_iters} iterations...")
    current_solver.train(warmup_iters)
    logger.info("Pre-training complete. Starting server...")
    
    api_host = config.get("api_host", "0.0.0.0")
    api_port = config.get("api_port", 8000)
    logger.info(f"Server running on http://{api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port)