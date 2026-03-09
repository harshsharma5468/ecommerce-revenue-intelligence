"""Database module for e-commerce revenue intelligence."""
from .db_manager import DEFAULT_DB, DatabaseManager, get_db

__all__ = ["DatabaseManager", "get_db", "DEFAULT_DB"]
