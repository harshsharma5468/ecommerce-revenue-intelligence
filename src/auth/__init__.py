"""Authentication module for dashboard access control."""
from .auth import USER_ROLES, AuthManager, auth_manager, get_auth_manager

__all__ = ["AuthManager", "get_auth_manager", "auth_manager", "USER_ROLES"]
