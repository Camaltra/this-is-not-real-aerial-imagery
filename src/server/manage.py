from flask.cli import FlaskGroup

from project import app, engine, Session
from project.models import Base
from project.models.user import User

cli = FlaskGroup(app)


@cli.command("create_db")
def create_db():
    with app.app_context():
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)


@cli.command("seed_db")
def seed_db():
    with app.app_context():
        session = Session()
        user = User(email="michael@mherman.org")
        session.add(user)
        session.commit()


if __name__ == "__main__":
    cli()
