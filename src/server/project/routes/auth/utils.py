from project.extension import bcrypt

import logging

logger = logging.getLogger("PasswordUtils")
logger.setLevel(logging.WARNING)


def compute_password(password: str) -> str:
    try:
        hashed_password = bcrypt.generate_password_hash(password)
        return hashed_password.decode("utf8")
    except Exception as e:
        logger.error(f"Error while hashing the password: {e}")
        raise ValueError("Error while hashing the password") from e
