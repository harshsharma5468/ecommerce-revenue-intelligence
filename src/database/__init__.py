"""Database module for e-commerce revenue intelligence."""
from .db_manager import DatabaseManager, get_db, DEFAULT_DB

__all__ = ["DatabaseManager", "get_db", "DEFAULT_DB"]
