"""
Strategy persistence utilities for saving and loading solver solutions.
Allows caching of solutions to avoid re-solving identical boards.
"""
import os
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StrategyCache:
    """Handles saving and loading of CFR solutions."""
    
    CACHE_DIR = "strategy_cache"
    
    @classmethod
    def _ensure_cache_dir(cls) -> None:
        """Ensure cache directory exists."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
    
    @classmethod
    def _board_to_filename(cls, board_cards: list[str]) -> str:
        """
        Convert board cards to a cache filename.
        
        Args:
            board_cards: List of card strings
            
        Returns:
            Filename for caching
        """
        board_str = "_".join(board_cards)
        return os.path.join(cls.CACHE_DIR, f"strategy_{board_str}.json")
    
    @classmethod
    def save_strategy(cls, board_cards: list[str], solver_data: Dict[str, Any]) -> bool:
        """
        Save a trained strategy to disk.
        
        Args:
            board_cards: List of board cards
            solver_data: Strategy data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cls._ensure_cache_dir()
            filepath = cls._board_to_filename(board_cards)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {
                "board": board_cards,
                "strategy_sum": solver_data["strategy_sum"].tolist() if isinstance(solver_data.get("strategy_sum"), np.ndarray) else solver_data.get("strategy_sum"),
                "regret_sum": solver_data["regret_sum"].tolist() if isinstance(solver_data.get("regret_sum"), np.ndarray) else solver_data.get("regret_sum"),
                "node_names": solver_data.get("node_names", {}),
                "actions": solver_data.get("actions", {}),
                "hands": solver_data.get("hands", []),
            }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Strategy cached for board: {' '.join(board_cards)}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to save strategy: {str(e)}")
            return False
    
    @classmethod
    def load_strategy(cls, board_cards: list[str]) -> Optional[Dict[str, Any]]:
        """
        Load a cached strategy from disk.
        
        Args:
            board_cards: List of board cards
            
        Returns:
            Strategy data if found, None otherwise
        """
        try:
            filepath = cls._board_to_filename(board_cards)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            if "strategy_sum" in data and isinstance(data["strategy_sum"], list):
                data["strategy_sum"] = np.array(data["strategy_sum"], dtype=np.float32)
            if "regret_sum" in data and isinstance(data["regret_sum"], list):
                data["regret_sum"] = np.array(data["regret_sum"], dtype=np.float32)
            
            logger.info(f"Strategy loaded from cache for board: {' '.join(board_cards)}")
            return data
        
        except Exception as e:
            logger.warning(f"Failed to load strategy: {str(e)}")
            return None
    
    @classmethod
    def clear_cache(cls) -> bool:
        """
        Clear all cached strategies.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(cls.CACHE_DIR):
                import shutil
                shutil.rmtree(cls.CACHE_DIR)
                os.makedirs(cls.CACHE_DIR, exist_ok=True)
                logger.info("Strategy cache cleared")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to clear cache: {str(e)}")
            return False
    
    @classmethod
    def list_cached_boards(cls) -> list[str]:
        """
        List all cached board strategies.
        
        Returns:
            List of board card combinations stored in cache
        """
        try:
            cls._ensure_cache_dir()
            cached = []
            for filename in os.listdir(cls.CACHE_DIR):
                if filename.startswith("strategy_") and filename.endswith(".json"):
                    # Extract board from filename
                    board_str = filename.replace("strategy_", "").replace(".json", "")
                    board = board_str.split("_")
                    if len(board) == 5:
                        cached.append(board)
            return cached
        except Exception as e:
            logger.warning(f"Failed to list cached boards: {str(e)}")
            return []
