"""Authentication module for dashboard access control."""
from .auth import AuthManager, get_auth_manager, auth_manager, USER_ROLES

__all__ = ["AuthManager", "get_auth_manager", "auth_manager", "USER_ROLES"]
