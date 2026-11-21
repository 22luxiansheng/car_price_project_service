"""SQLite 持久化：保存每次预测，并支持回填真实成交价。"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    make: Mapped[str] = mapped_column(String(50))
    yr_mfr: Mapped[int] = mapped_column(Integer)
    kms_run: Mapped[float] = mapped_column(Float)
    fuel_type: Mapped[str] = mapped_column(String(30))
    transmission: Mapped[str] = mapped_column(String(30))
    body_type: Mapped[str] = mapped_column(String(30))
    predicted_price: Mapped[float] = mapped_column(Float)
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    with SessionLocal() as session:
        yield session
