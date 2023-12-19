from project.extension import bcrypt


def compute_pasword(password: str) -> str:
    return bcrypt.generate_password_hash(password).decode("utf8")
