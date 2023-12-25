from flask.cli import FlaskGroup

from project import app, engine, Session
from project.models.base import Base
from project.models.user import User
from project.routes.auth.utils import compute_password

cli = FlaskGroup(app)


@cli.command("create_db")
def create_db():
    with app.app_context():
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)


@cli.command("create_admin")
def create_admin():
    with app.app_context():
        session = Session()
        user = User(
            email="admin@admin.com",
            password=compute_password("admin"),
            username="Admin",
            admin=True,
        )
        session.add(user)
        session.commit()


if __name__ == "__main__":
    cli()
