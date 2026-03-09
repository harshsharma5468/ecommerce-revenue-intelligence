"""
User Authentication Module
Simple authentication system for the dashboard.

Usage:
    from src.auth import AuthManager
    
    auth = AuthManager()
    auth.add_user("admin", "password123")  # Add users
    # In dashboard callback:
    if not auth.is_authenticated():
        return login_layout()
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Default users (in production, use environment variables or secure storage)
DEFAULT_USERS = {
    "admin": "admin123",
    "analyst": "analyst123",
    "viewer": "viewer123",
}

# User roles with permissions
USER_ROLES = {
    "admin": {"view": True, "export": True, "settings": True, "admin": True},
    "analyst": {"view": True, "export": True, "settings": False, "admin": False},
    "viewer": {"view": True, "export": False, "settings": False, "admin": False},
}


class AuthManager:
    """
    Simple authentication manager for dashboard access control.
    """
    
    def __init__(self, users_file: str = "users.json"):
        """
        Initialize authentication manager.
        
        Args:
            users_file: Path to store user credentials (JSON)
        """
        self.users_file = Path(users_file)
        self.sessions: Dict[str, dict] = {}
        self.users = self._load_users()
        
        # Add default users if none exist
        if not self.users:
            for username, password in DEFAULT_USERS.items():
                self.add_user(username, password, "admin" if username == "admin" else "viewer")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{hashed}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, hashed = stored_hash.split(":")
            check_hash = hashlib.sha256((salt + password).encode()).hexdigest()
            return check_hash == hashed
        except Exception:
            return False
    
    def _load_users(self) -> Dict:
        """Load users from JSON file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading users: {e}")
        return {}
    
    def _save_users(self):
        """Save users to JSON file."""
        try:
            with open(self.users_file, "w") as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def add_user(self, username: str, password: str, role: str = "viewer") -> bool:
        """
        Add a new user.
        
        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            role: User role (admin, analyst, viewer)
            
        Returns:
            True if user added successfully
        """
        if username in self.users:
            return False
        
        self.users[username] = {
            "password": self._hash_password(password),
            "role": role,
            "created": datetime.now().isoformat(),
            "permissions": USER_ROLES.get(role, USER_ROLES["viewer"]),
        }
        self._save_users()
        logger.info(f"User added: {username} (role: {role})")
        return True
    
    def remove_user(self, username: str) -> bool:
        """Remove a user."""
        if username in self.users and username != "admin":
            del self.users[username]
            self._save_users()
            return True
        return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and create session.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Session token if successful, None otherwise
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: unknown user {username}")
            return None
        
        if not self._verify_password(password, self.users[username]["password"]):
            logger.warning(f"Authentication failed: wrong password for {username}")
            return None
        
        # Create session
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            "username": username,
            "role": self.users[username]["role"],
            "permissions": self.users[username]["permissions"],
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=24),
        }
        
        logger.info(f"User authenticated: {username}")
        return token
    
    def validate_session(self, token: str) -> Optional[dict]:
        """
        Validate session token.
        
        Args:
            token: Session token
            
        Returns:
            Session info if valid, None otherwise
        """
        if token not in self.sessions:
            return None
        
        session = self.sessions[token]
        if datetime.now() > session["expires"]:
            del self.sessions[token]
            return None
        
        return session
    
    def logout(self, token: str):
        """Invalidate session token."""
        if token in self.sessions:
            del self.sessions[token]
    
    def get_user_permissions(self, token: str) -> Dict:
        """Get permissions for authenticated user."""
        session = self.validate_session(token)
        if session:
            return session.get("permissions", {})
        return {"view": False, "export": False, "settings": False, "admin": False}
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        if username not in self.users:
            return False
        
        if not self._verify_password(old_password, self.users[username]["password"]):
            return False
        
        self.users[username]["password"] = self._hash_password(new_password)
        self._save_users()
        return True
    
    def list_users(self) -> list:
        """List all usernames (without passwords)."""
        return [
            {
                "username": username,
                "role": data["role"],
                "created": data.get("created", "unknown"),
            }
            for username, data in self.users.items()
        ]


# Global auth manager instance
auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance."""
    return auth_manager
