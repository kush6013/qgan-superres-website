from datetime import datetime, timezone, timedelta
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean
from database import Base


def get_ist_now():
    """Get current time in Indian Standard Time (IST, UTC+5:30)"""
    utc_now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    return utc_now + ist_offset


class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    result_path = Column(String, nullable=False)
    dicom_path = Column(String, nullable=True)
    description = Column(String, nullable=True)
    tags = Column(String, nullable=True)
    patient_id = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    sharpness = Column(Float, nullable=True)
    slice_index = Column(Integer, nullable=True)
    window_preset = Column(String, nullable=True)
    scale_factor = Column(Integer, default=8)
    deidentified = Column(Boolean, default=False)
    modality = Column(String, default="general", nullable=True)
    timestamp = Column(DateTime, default=get_ist_now)

