from datetime import datetime, timezone, timedelta
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from database import Base


def get_ist_now():
    """Get current time in Indian Standard Time (IST, UTC+5:30)"""
    utc_now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    return utc_now + ist_offset


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String, default="processing", nullable=False)  # processing, completed, error
    progress = Column(Integer, default=5, nullable=False)
    step = Column(String, default="Initializing...", nullable=False)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)
