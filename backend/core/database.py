from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.core.config import settings

SQLALCHEMY_DATABASE_URL = f"sqlite:///{settings.DATABASE_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def configure_sqlite(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
HISTORY_IMAGE_DIR = settings.HISTORY_IMAGE_DIR


def initialize_database(database_engine=engine):
    """Create the current schema and add history columns to existing databases."""
    Base.metadata.create_all(bind=database_engine)

    inspector = inspect(database_engine)
    if "caption_records" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("caption_records")
    }
    with database_engine.begin() as connection:
        if "image_path" not in existing_columns:
            connection.execute(
                text("ALTER TABLE caption_records ADD COLUMN image_path VARCHAR")
            )
        if "result_json" not in existing_columns:
            connection.execute(
                text("ALTER TABLE caption_records ADD COLUMN result_json TEXT")
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
