from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enables SQLite WAL (Write-Ahead Logging) mode and busy timeout for safe concurrent access."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)
Base = declarative_base()


def init_db() -> None:
    from models.result import Result
    from models.user import User
    from models.job import Job

    Base.metadata.create_all(bind=engine)
    _create_missing_columns()


def _create_missing_columns() -> None:
    migrations = {
        "results": {
            "description": "TEXT",
            "tags": "TEXT",
            "patient_id": "TEXT",
            "owner_id": "INTEGER",
            "timestamp": "DATETIME",
            "psnr": "REAL",
            "ssim": "REAL",
            "sharpness": "REAL",
            "slice_index": "INTEGER",
            "window_preset": "TEXT",
            "scale_factor": "INTEGER",
            "dicom_path": "TEXT",
            "deidentified": "BOOLEAN",
            "modality": "TEXT",
        },
        "users": {
            "is_admin": "BOOLEAN",
            "reset_token": "TEXT",
            "reset_token_expires": "DATETIME",
            "created_at": "DATETIME",
        },
    }

    with engine.begin() as conn:
        for table_name, cols in migrations.items():
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            existing_columns = {row[1] for row in result.fetchall()}
            for column_name, column_type in cols.items():
                if column_name not in existing_columns:
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    ))
                    print(f"Added missing column {column_name} to {table_name}")
