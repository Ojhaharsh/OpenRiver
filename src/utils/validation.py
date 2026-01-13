"""
Input validation utilities for poker solver.
Handles board card validation and error reporting.
"""
from typing import List, Tuple


def validate_board_cards(board_str: str) -> Tuple[bool, str, List[str] | None]:
    """
    Validate board card input string.
    
    Args:
        board_str: Space-separated card string (e.g., "Ks Th 7s 4d 2s")
    
    Returns:
        Tuple of (is_valid: bool, error_message: str, cards: List[str] or None)
    """
    if not board_str or not isinstance(board_str, str):
        return False, "Board input must be a non-empty string", None
    
    board_str = board_str.strip()
    cards = board_str.split()
    
    # Must have exactly 5 cards
    if len(cards) != 5:
        return False, f"Expected 5 board cards, got {len(cards)}", None
    
    valid_ranks = set('23456789TJQKA')
    valid_suits = set('shdc')
    
    # Validate each card format
    seen = set()
    for card in cards:
        if len(card) != 2:
            return False, f"Invalid card format '{card}'. Use format like 'As', 'Kh', etc.", None
        
        rank, suit = card[0], card[1]
        
        if rank not in valid_ranks:
            return False, f"Invalid rank '{rank}' in card '{card}'. Must be 2-9, T, J, Q, K, A", None
        
        if suit not in valid_suits:
            return False, f"Invalid suit '{suit}' in card '{card}'. Must be s, h, d, or c", None
        
        # Check for duplicates
        if card in seen:
            return False, f"Duplicate card '{card}' in board", None
        seen.add(card)
    
    return True, "", cards


def validate_iterations(iters: int) -> Tuple[bool, str]:
    """
    Validate iteration count.
    
    Args:
        iters: Number of training iterations
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(iters, int):
        return False, "Iterations must be an integer"
    
    if iters <= 0:
        return False, "Iterations must be greater than 0"
    
    if iters > 1_000_000:
        return False, "Iterations must not exceed 1,000,000 (too expensive)"
    
    return True, ""
