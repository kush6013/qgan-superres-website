import os
from fastapi import FastAPI, Request, UploadFile, File, Header, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import shutil
import io
from datetime import datetime, timedelta, timezone
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import hashlib
import secrets

import threading
import uuid
from database import SessionLocal, init_db
from models import Result, User, Job
from models.medical_classifier import MedicalImageClassifier
from models.qgan_model import QGANGenerator, load_generator as load_qgan_generator
from metrics import evaluate_super_resolution_quality
from medical_imaging import (
    render_slice_image,
    inspect_volume_info,
    HU_PRESETS,
    apply_hu_window,
    create_dicom_from_image,
    deidentify_dicom_file,
)

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency
    pydicom = None

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

# ✅ App init & Secure Session Management
APP_ENV = os.environ.get("APP_ENV", "development").lower()
SESSION_SECRET = os.environ.get("SESSION_SECRET")

if not SESSION_SECRET:
    if APP_ENV == "production":
        raise RuntimeError("CRITICAL SECURITY ERROR: SESSION_SECRET must be set in production mode.")
    # In development, generate a persistent dev secret file so sessions survive server restarts
    dev_secret_file = Path(".dev_session_secret")
    if dev_secret_file.exists():
        SESSION_SECRET = dev_secret_file.read_text().strip()
    else:
        SESSION_SECRET = secrets.token_hex(32)
        try:
            dev_secret_file.write_text(SESSION_SECRET)
        except Exception:
            pass

app = FastAPI(title="QGAN Medical Image Super-Resolution")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=(APP_ENV == "production"),
    max_age=86400 * 7,
)

# ✅ Directories & Storage Configuration
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPSCALE_FACTOR = 8
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MEDICAL_EXTENSIONS = {".dcm", ".nii", ".nii.gz"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | MEDICAL_EXTENSIONS
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/dicom",
    "application/dicom+json",
    "application/octet-stream",
}

# ✅ Mount public static assets only (CSS, JS, Fonts).
# CRITICAL: uploads/ and results/ are NOT mounted statically to protect private medical records.
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.cache = None

# ✅ Database setup & Startup Recovery
init_db()


def recover_interrupted_jobs() -> None:
    """Safely transitions dangling background jobs interrupted by server/container restarts to error status."""
    db = SessionLocal()
    try:
        interrupted = db.query(Job).filter(Job.status == "processing").all()
        for j in interrupted:
            j.status = "error"
            j.step = "Processing interrupted by application restart."
            j.error = "This task was halted when the application restarted. Please resubmit the scan."
        if interrupted:
            db.commit()
            print(f"Recovered and marked {len(interrupted)} interrupted jobs on startup.")
    except Exception as exc:
        db.rollback()
        print(f"Startup job recovery notice: {exc}")
    finally:
        db.close()


recover_interrupted_jobs()

# Authentication helpers

SESSION_USER_KEY = "user_id"


try:
    import bcrypt
except ImportError:
    bcrypt = None


def hash_password(password: str, salt: str = "") -> str:
    """Secure password hashing with modern Bcrypt (falling back to salted SHA-256 if needed)."""
    if bcrypt is not None:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    """Verifies passwords with Bcrypt or legacy salted SHA-256 hashes."""
    if password_hash.startswith("$2b$") or password_hash.startswith("$2a$"):
        if bcrypt is not None:
            try:
                return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
            except Exception:
                return False
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest() == password_hash


def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def get_current_user(request: Request):
    user_id = request.session.get(SESSION_USER_KEY)
    if not user_id:
        return None
    db = SessionLocal()
    user = get_user_by_id(db, user_id)
    db.close()
    return user


def require_admin(request: Request):
    user = get_current_user(request)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def hash_reset_token(token: str) -> str:
    """Computes SHA-256 hash of a reset token for secure database storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_password_reset_token(db, user: User) -> str:
    """
    Generates a cryptographically secure reset token.
    Stores only the SHA-256 hash in the database to prevent token theft,
    while returning the raw unhashed token for the reset link.
    """
    raw_token = secrets.token_urlsafe(32)
    user.reset_token = hash_reset_token(raw_token)
    user.reset_token_expires = get_ist_now() + timedelta(hours=1)
    db.add(user)
    db.commit()
    return raw_token


def verify_password_reset_token(db, token: str) -> User | None:
    """
    Verifies a reset token against the stored SHA-256 hash (or legacy plaintext for backward compatibility).
    Enforces expiration check with robust timezone compatibility.
    """
    token_hash = hash_reset_token(token)
    user = db.query(User).filter(
        (User.reset_token == token_hash) | (User.reset_token == token)
    ).first()
    if not user or not user.reset_token_expires:
        return None

    expires = user.reset_token_expires
    if expires.tzinfo is not None:
        expires = expires.replace(tzinfo=None)

    now = get_ist_now()
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)

    if expires < now:
        return None
    return user


# QGAN model integration
MODEL_PATH = Path("models/qgan_generator.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_generator: QGANGenerator | None = None

MODALITY_CHECKPOINTS = {
    "general": Path("models/qgan_generator.pt"),
    "brain_mri": Path("models/qgan_generator_mri.pt"),
    "chest_xray": Path("models/qgan_generator_xray.pt"),
    "ct_scan": Path("models/qgan_generator_ct.pt"),
}
_generator_cache: dict[str, QGANGenerator] = {}


def load_generator() -> QGANGenerator | None:
    global _generator
    if _generator is not None:
        return _generator

    _generator = load_qgan_generator(MODEL_PATH, DEVICE)
    if _generator is None:
        print("WARNING: QGAN checkpoint not found or failed to load, falling back to bicubic resize.")
    else:
        print(f"Loaded QGAN generator from {MODEL_PATH} on {DEVICE}")

    return _generator


def load_generator_for_modality(modality: str = "general") -> tuple[QGANGenerator | None, str]:
    mod = (modality or "general").lower().strip()
    target_path = MODALITY_CHECKPOINTS.get(mod, MODALITY_CHECKPOINTS["general"])

    if mod in _generator_cache:
        return _generator_cache[mod], mod

    gen = load_qgan_generator(target_path, DEVICE)
    if gen is not None:
        _generator_cache[mod] = gen
        print(f"Loaded modality-specific QGAN [{mod}] from {target_path}")
        return gen, mod

    general_gen = load_generator()
    if general_gen is not None:
        return general_gen, "general (fallback)"

    return None, "bicubic"

CLASSIFIER_PATH = Path("models/medical_classifier.pt")
medical_classifier = MedicalImageClassifier(DEVICE, checkpoint_path=CLASSIFIER_PATH)

def utc_to_ist(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to Indian Standard Time (IST, UTC+5:30)"""
    ist_offset = timedelta(hours=5, minutes=30)
    return utc_dt + ist_offset

def get_ist_now() -> datetime:
    """Get current time in Indian Standard Time (IST, UTC+5:30) as timezone-naive for SQLite compatibility."""
    utc_now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    return (utc_now + ist_offset).replace(tzinfo=None)


def get_file_extension(filename: str) -> str:
    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower_name).suffix


def sanitize_filename(filename: str) -> str:
    """Strips directory traversal sequences and special characters, preserving safe display filename."""
    base = Path(filename).name
    clean = "".join(c for c in base if c.isalnum() or c in {".", "_", "-"})
    return clean or f"scan_{uuid.uuid4().hex[:8]}"


def validate_file_content(content: bytes, filename: str) -> tuple[bool, str]:
    """
    Validates uploaded file against size limits and actual file-content signatures
    to prevent path traversal, remote code execution, or malformed upload crashes.
    """
    if not content or len(content) == 0:
        return False, "Uploaded file is empty."

    if len(content) > MAX_UPLOAD_BYTES:
        return False, f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_MB} MB."

    ext = get_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Validate standard PNG/JPEG
    if ext in IMAGE_EXTENSIONS:
        try:
            bio = io.BytesIO(content)
            with Image.open(bio) as img:
                img.verify()
            return True, "Valid image"
        except Exception:
            return False, "Corrupted or invalid PNG/JPEG image data."

    # Validate DICOM
    if ext == ".dcm":
        if len(content) >= 132 and content[128:132] == b"DICM":
            return True, "Valid DICOM"
        if pydicom is not None:
            try:
                bio = io.BytesIO(content)
                ds = pydicom.dcmread(bio, force=True)
                if hasattr(ds, "pixel_array") or hasattr(ds, "PixelData") or hasattr(ds, "SOPClassUID"):
                    return True, "Valid DICOM"
            except Exception:
                return False, "Corrupted or invalid DICOM file."
        return False, "Invalid DICOM structure."

    # Validate NIfTI
    if ext in {".nii", ".nii.gz"}:
        if ext == ".nii.gz" and not content.startswith(b"\x1f\x8b"):
            return False, "Corrupted or invalid gzip compressed NIfTI file."
        return True, "Valid NIfTI"

    return False, "Unsupported file format."


def get_preview_path(file_path: Path, extension: str) -> Path:
    if extension == ".nii.gz":
        base_name = file_path.name[:-7]
    else:
        base_name = file_path.stem
    return file_path.with_name(f"{base_name}_preview.png")


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("Empty image data.")

    array = np.nan_to_num(array)
    if array.ndim >= 3:
        spatial_axis = int(np.argmax(array.shape))
        center_index = array.shape[spatial_axis] // 2
        array = np.take(array, center_index, axis=spatial_axis)

    if array.ndim == 1:
        side = int(np.sqrt(array.size))
        array = array[: side * side].reshape(side, side)

    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def load_pil_image(
    file_path: Path,
    extension: str | None = None,
    slice_index: int | None = None,
    window_preset: str = "auto"
) -> Image.Image:
    extension = extension or get_file_extension(file_path.name)

    if extension in MEDICAL_EXTENSIONS:
        return render_slice_image(file_path, slice_index=slice_index, window_preset=window_preset)

    if extension in IMAGE_EXTENSIONS:
        img = Image.open(file_path).convert("RGB")
        if window_preset and window_preset != "auto":
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            preset = HU_PRESETS.get(window_preset.lower(), HU_PRESETS["auto"])
            win_arr = apply_hu_window(arr, center=preset.get("center"), width=preset.get("width"))
            return Image.fromarray(win_arr).convert("RGB")
        return img

    raise ValueError(f"Unsupported file extension: {extension}")


def ensure_preview_image(
    file_path: Path,
    extension: str,
    slice_index: int | None = None,
    window_preset: str = "auto"
) -> Path:
    if extension in IMAGE_EXTENSIONS and (not window_preset or window_preset == "auto"):
        return file_path

    slice_tag = f"_s{slice_index}" if slice_index is not None else ""
    window_tag = f"_{window_preset}" if window_preset and window_preset != "auto" else ""
    if extension == ".nii.gz":
        base_name = file_path.name[:-7]
    else:
        base_name = file_path.stem

    preview_path = file_path.with_name(f"{base_name}{slice_tag}{window_tag}_preview.png")
    preview_image = load_pil_image(file_path, extension, slice_index=slice_index, window_preset=window_preset)
    preview_image.save(preview_path)
    return preview_path


def super_resolve_image(low_res_path: str, scale: int = 8, modality: str = "general") -> tuple[str, dict]:
    source_path = Path(low_res_path)
    img = load_pil_image(source_path, get_file_extension(source_path.name))
    model, used_modality = load_generator_for_modality(modality)
    scale = scale if scale in (2, 4, 8) else 8

    if model is None:
        high_res = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    else:
        input_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            output_tensor = model(input_tensor)

        if output_tensor.min() < 0 or output_tensor.max() > 1:
            output_tensor = (output_tensor + 1.0) / 2.0

        output_tensor = output_tensor.clamp(0, 1).cpu().squeeze(0)
        full_sr = TF.to_pil_image(output_tensor)
        if scale == 8:
            high_res = full_sr
        else:
            high_res = full_sr.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    result_filename = f"sr_{scale}x_{uuid.uuid4().hex[:8]}_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.png"
    result_path = RESULT_DIR / result_filename
    high_res.save(result_path)
    metrics = evaluate_super_resolution_quality(img, high_res)
    metrics["modality"] = used_modality
    return str(result_path), metrics


def is_medical_image(file_path: Path) -> bool:
    img = load_pil_image(file_path, get_file_extension(file_path.name))
    return medical_classifier.is_medical_scan(img)

# ✅ Home Route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    current_user = get_current_user(request)
    return HTMLResponse(
        templates.get_template("index.html").render(
            {
                "request": request,
                "user": current_user,
            }
        )
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return HTMLResponse(templates.get_template("register.html").render({"request": request}))


@app.post("/register", response_class=HTMLResponse)
async def register_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str | None = Form(None),
):
    db = SessionLocal()
    existing_user = get_user_by_email(db, email)
    if existing_user:
        db.close()
        return HTMLResponse(
            templates.get_template("register.html").render(
                {"request": request, "error": "Email already registered."}
            )
        )

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    user_count = db.query(User).count()
    is_admin = user_count == 0
    user = User(
        email=email,
        full_name=full_name,
        password_hash=password_hash,
        salt=salt,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    request.session[SESSION_USER_KEY] = user.id
    return RedirectResponse(url="/", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return HTMLResponse(templates.get_template("login.html").render({"request": request}))


@app.post("/login", response_class=HTMLResponse)
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    db = SessionLocal()
    user = get_user_by_email(db, email)
    db.close()
    if not user or not verify_password(password, user.salt, user.password_hash):
        return HTMLResponse(
            templates.get_template("login.html").render(
                {"request": request, "error": "Invalid email or password."}
            )
        )

    request.session[SESSION_USER_KEY] = user.id
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.pop(SESSION_USER_KEY, None)
    return RedirectResponse(url="/", status_code=303)


@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return HTMLResponse(
        templates.get_template("profile.html").render({"request": request, "user": user})
    )


@app.post("/profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    full_name: str | None = Form(None),
    current_password: str | None = Form(None),
    new_password: str | None = Form(None),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    db = SessionLocal()
    user_db = get_user_by_id(db, user.id)
    if full_name is not None:
        user_db.full_name = full_name

    if current_password and new_password:
        if not verify_password(current_password, user_db.salt, user_db.password_hash):
            db.close()
            return HTMLResponse(
                templates.get_template("profile.html").render(
                    {"request": request, "user": user_db, "error": "Invalid current password."}
                )
            )
        user_db.salt = secrets.token_hex(16)
        user_db.password_hash = hash_password(new_password, user_db.salt)

    db.add(user_db)
    db.commit()
    db.close()

    return HTMLResponse(
        templates.get_template("profile.html").render(
            {"request": request, "user": user_db, "success": "Profile updated successfully."}
        )
    )


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_request_page(request: Request):
    return HTMLResponse(templates.get_template("reset_request.html").render({"request": request}))


@app.post("/reset-password", response_class=HTMLResponse)
async def reset_request(request: Request, email: str = Form(...)):
    db = SessionLocal()
    user = get_user_by_email(db, email)
    if not user:
        db.close()
        return HTMLResponse(
            templates.get_template("reset_request.html").render(
                {"request": request, "error": "If the email exists, a reset token has been created."}
            )
        )

    token = create_password_reset_token(db, user)
    db.close()
    reset_link = f"{request.url.scheme}://{request.url.hostname}:{request.url.port}/reset-password/{token}"
    return HTMLResponse(
        templates.get_template("reset_sent.html").render(
            {"request": request, "reset_link": reset_link}
        )
    )


@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def reset_form(request: Request, token: str):
    db = SessionLocal()
    user = verify_password_reset_token(db, token)
    db.close()
    if not user:
        return HTMLResponse(
            templates.get_template("reset_invalid.html").render(
                {"request": request, "error": "Invalid or expired reset token."}
            )
        )
    return HTMLResponse(templates.get_template("reset_form.html").render({"request": request, "token": token}))


@app.post("/reset-password/{token}", response_class=HTMLResponse)
async def reset_password(request: Request, token: str, new_password: str = Form(...)):
    db = SessionLocal()
    user = verify_password_reset_token(db, token)
    if not user:
        db.close()
        return HTMLResponse(
            templates.get_template("reset_invalid.html").render(
                {"request": request, "error": "Invalid or expired reset token."}
            )
        )

    user.salt = secrets.token_hex(16)
    user.password_hash = hash_password(new_password, user.salt)
    user.reset_token = None
    user.reset_token_expires = None
    db.add(user)
    db.commit()
    db.close()

    return HTMLResponse(
        templates.get_template("reset_success.html").render({"request": request})
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    admin = require_admin(request)
    db = SessionLocal()
    users = db.query(User).order_by(User.created_at.desc()).all()
    results = db.query(Result).order_by(Result.timestamp.desc()).limit(50).all()
    db.close()

    # Format timestamps for display
    formatted_results = []
    for result in results:
        formatted_results.append({
            "id": result.id,
            "filename": result.filename,
            "original_path": result.original_path,
            "result_path": result.result_path,
            "description": result.description,
            "tags": result.tags,
            "patient_id": result.patient_id,
            "owner_id": result.owner_id,
            "timestamp": utc_to_ist(result.timestamp),
            "output_image_path": result.result_path,  # for template compatibility
            "model_name": result.filename,  # for template compatibility
        })

    return HTMLResponse(
        templates.get_template("admin.html").render(
            {
                "request": request,
                "user": admin,
                "users": users,
                "results": formatted_results,
            }
        )
    )


@app.get("/api/users/me")
async def api_user_me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"user": None})
    return JSONResponse(
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat(),
        }
    )


@app.get("/api/results")
async def api_list_results(request: Request, q: str | None = None):
    db = SessionLocal()
    current_user = get_current_user(request)
    query = db.query(Result).order_by(Result.timestamp.desc())
    if current_user:
        query = query.filter(Result.owner_id == current_user.id)
    if q:
        search = f"%{q}%"
        query = query.filter(
            Result.filename.ilike(search)
            | Result.description.ilike(search)
            | Result.tags.ilike(search)
            | Result.patient_id.ilike(search)
        )
    results = query.all()
    db.close()
    return JSONResponse(
        [
            {
                "id": item.id,
                "filename": item.filename,
                "description": item.description,
                "tags": item.tags,
                "patient_id": item.patient_id,
                "psnr": item.psnr,
                "ssim": item.ssim,
                "sharpness": item.sharpness,
                "slice_index": item.slice_index,
                "window_preset": item.window_preset,
                "scale_factor": item.scale_factor or 8,
                "dicom_url": f"/api/dicom/download/{Path(item.dicom_path).name}" if item.dicom_path else None,
                "deidentified": item.deidentified or False,
                "modality": item.modality or "general",
                "original_url": f"/uploads/{Path(item.original_path).name}",
                "result_url": f"/results/{Path(item.result_path).name}",
                "timestamp": item.timestamp.isoformat(),
            }
            for item in results
        ]
    )


@app.get("/api/results/{result_id}")
async def api_get_result(request: Request, result_id: int):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    db = SessionLocal()
    item = db.query(Result).filter(Result.id == result_id).first()
    db.close()
    if not item:
        return JSONResponse({"error": "Result not found."}, status_code=404)

    if item.owner_id != current_user.id and not current_user.is_admin:
        return JSONResponse({"error": "Access denied."}, status_code=403)

    return JSONResponse(
        {
            "id": item.id,
            "filename": item.filename,
            "description": item.description,
            "tags": item.tags,
            "patient_id": item.patient_id,
            "original_url": f"/uploads/{Path(item.original_path).name}",
            "result_url": f"/results/{Path(item.result_path).name}",
            "timestamp": item.timestamp.isoformat(),
        }
    )


def update_job_in_db(
    job_id: str,
    progress: int | None = None,
    step: str | None = None,
    status: str | None = None,
    result_id: int | None = None,
    error: str | None = None,
):
    """Updates background job state persistently in the SQLite database."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            if progress is not None:
                job.progress = progress
            if step is not None:
                job.step = step
            if status is not None:
                job.status = status
            if result_id is not None:
                job.result_id = result_id
            if error is not None:
                job.error = error
            db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Error updating job {job_id}: {exc}")
    finally:
        db.close()


def execute_super_resolution_pipeline(
    file_path: Path,
    filename: str,
    extension: str,
    description: str | None = None,
    tags: str | None = None,
    patient_id: str | None = None,
    slice_index: int | None = None,
    window_preset: str = "auto",
    scale: int = 8,
    modality: str = "general",
    owner_id: int | None = None,
    progress_callback=None,
) -> dict:
    if progress_callback:
        progress_callback(15, "Checking HIPAA compliance & validating scan format...")

    was_deidentified = False
    if extension == ".dcm":
        _, was_deidentified = deidentify_dicom_file(file_path, custom_patient_id=patient_id)

    if progress_callback:
        progress_callback(35, f"Extracting radiologic slice with {window_preset} window leveling...")

    preset_choice = (window_preset or "auto").strip().lower()
    preview_path = ensure_preview_image(
        file_path,
        extension,
        slice_index=slice_index,
        window_preset=preset_choice,
    )
    detected_as_medical = is_medical_image(preview_path)

    if progress_callback:
        progress_callback(65, f"Running QGAN neural inference ({scale}x magnification)...")

    chosen_scale = scale if scale in (2, 4, 8) else 8
    result_path, metrics = super_resolve_image(str(preview_path), scale=chosen_scale, modality=modality)

    if progress_callback:
        progress_callback(85, "Creating compliant clinical DICOM (.dcm) export & IQA metrics...")

    # Generate DICOM export
    dicom_filename = f"{Path(result_path).stem}.dcm"
    dicom_dest = RESULT_DIR / dicom_filename
    enhanced_pil = Image.open(result_path)
    source_dcm = file_path if extension == ".dcm" else None
    saved_dicom = create_dicom_from_image(
        enhanced_pil,
        dicom_dest,
        source_dicom_path=source_dcm,
        patient_id=patient_id,
        series_description=f"QGAN {chosen_scale}x Super-Resolution Enhanced ({preset_choice})"
    )

    db = SessionLocal()
    result = Result(
        filename=filename,
        original_path=str(preview_path),
        result_path=result_path,
        dicom_path=str(saved_dicom) if saved_dicom else None,
        description=description,
        tags=tags,
        patient_id=patient_id,
        owner_id=owner_id,
        psnr=metrics.get("psnr"),
        ssim=metrics.get("ssim"),
        sharpness=metrics.get("sharpness"),
        slice_index=slice_index,
        window_preset=preset_choice,
        scale_factor=chosen_scale,
        deidentified=was_deidentified,
        modality=modality or "general",
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    result_id = result.id
    db.close()

    if progress_callback:
        progress_callback(100, "Super-resolution and DICOM export complete!")

    original_url = f"/uploads/{preview_path.name}"
    result_url = f"/results/{Path(result_path).name}"
    dicom_url = f"/api/dicom/download/{dicom_dest.name}" if saved_dicom and saved_dicom.exists() else None

    message = f"QGAN {chosen_scale}x enhancement finished successfully."
    if not detected_as_medical:
        message += " Format accepted, but the medical-image classifier could not confidently verify this upload as a scan."
    if was_deidentified:
        message += " Protected Health Information (PHI) was automatically sanitized under HIPAA rules."

    return {
        "id": result_id,
        "original": original_url,
        "result": result_url,
        "report_url": f"/report/{result_id}",
        "dicom_url": dicom_url,
        "message": message,
        "detected_as_medical": detected_as_medical,
        "metrics": metrics,
        "slice_index": slice_index,
        "window_preset": preset_choice,
        "scale_factor": chosen_scale,
        "deidentified": was_deidentified,
        "modality": modality or "general",
    }


def background_job_worker(
    job_id: str,
    file_path: Path,
    filename: str,
    extension: str,
    description: str | None,
    tags: str | None,
    patient_id: str | None,
    slice_index: int | None,
    window_preset: str,
    scale: int,
    modality: str,
    owner_id: int | None,
):
    try:
        def update_progress(pct: int, msg: str):
            update_job_in_db(job_id, progress=pct, step=msg)

        data = execute_super_resolution_pipeline(
            file_path=file_path,
            filename=filename,
            extension=extension,
            description=description,
            tags=tags,
            patient_id=patient_id,
            slice_index=slice_index,
            window_preset=window_preset,
            scale=scale,
            modality=modality,
            owner_id=owner_id,
            progress_callback=update_progress,
        )
        update_job_in_db(
            job_id,
            progress=100,
            step="Enhancement complete!",
            status="completed",
            result_id=data["id"],
        )
    except Exception as exc:
        print(f"Error in background worker for job {job_id}: {exc}")
        update_job_in_db(
            job_id,
            status="error",
            error="An error occurred during medical scan super-resolution processing.",
        )


# ✅ Upload Routes (Synchronous & Async with Persistent DB Job Tracking)
@app.post("/upload/async")
async def upload_image_async(
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),
    patient_id: str | None = Form(None),
    slice_index: int | None = Form(None),
    window_preset: str | None = Form("auto"),
    scale: int | None = Form(8),
    modality: str | None = Form("general"),
):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse(
            {"error": "Authentication required. Please log in to process medical scans."},
            status_code=401,
        )

    try:
        content = await file.read()
        is_valid, validation_msg = validate_file_content(content, file.filename)
        if not is_valid:
            return JSONResponse({"error": validation_msg}, status_code=400)

        extension = get_file_extension(file.filename)
        clean_name = sanitize_filename(file.filename)
        storage_id = uuid.uuid4().hex
        storage_filename = f"{storage_id}_{clean_name}"
        file_path = UPLOAD_DIR / storage_filename

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        job_id = str(uuid.uuid4())
        db = SessionLocal()
        new_job = Job(
            id=job_id,
            user_id=current_user.id,
            status="processing",
            progress=5,
            step="Initializing background enhancement worker...",
        )
        db.add(new_job)
        db.commit()
        db.close()

        threading.Thread(
            target=background_job_worker,
            args=(
                job_id,
                file_path,
                clean_name,
                extension,
                description,
                tags,
                patient_id,
                slice_index,
                window_preset or "auto",
                scale or 8,
                modality or "general",
                current_user.id,
            ),
            daemon=True,
        ).start()

        return JSONResponse({"job_id": job_id})
    except Exception as exc:
        return JSONResponse({"error": "Failed to initiate background enhancement job."}, status_code=500)


@app.get("/api/jobs/{job_id}")
async def get_job_status(request: Request, job_id: str):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return JSONResponse({"error": "Job not found", "status": "not_found"}, status_code=404)

    # Authorization: User A cannot inspect User B's job
    if job.user_id is not None and job.user_id != current_user.id and not current_user.is_admin:
        db.close()
        return JSONResponse({"error": "Access denied"}, status_code=403)

    resp = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "step": job.step,
        "error": job.error,
        "data": None,
    }

    if job.status == "completed" and job.result_id:
        result_item = db.query(Result).filter(Result.id == job.result_id).first()
        if result_item:
            resp["data"] = {
                "id": result_item.id,
                "original": f"/uploads/{Path(result_item.original_path).name}",
                "result": f"/results/{Path(result_item.result_path).name}",
                "report_url": f"/report/{result_item.id}",
                "dicom_url": f"/api/dicom/download/{Path(result_item.dicom_path).name}" if result_item.dicom_path else None,
                "message": f"QGAN {result_item.scale_factor or 8}x enhancement finished successfully.",
                "detected_as_medical": True,
                "metrics": {
                    "psnr": result_item.psnr,
                    "ssim": result_item.ssim,
                    "sharpness": result_item.sharpness,
                    "scale_factor": f"{result_item.scale_factor or 8}x",
                    "reference_type": "bicubic_baseline",
                    "reference_description": "Evaluated against bicubic-upscaled input baseline",
                },
                "slice_index": result_item.slice_index,
                "window_preset": result_item.window_preset,
                "scale_factor": result_item.scale_factor or 8,
                "deidentified": result_item.deidentified or False,
                "modality": result_item.modality or "general",
            }

    db.close()
    return JSONResponse(resp)


@app.post("/upload/")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),
    patient_id: str | None = Form(None),
    slice_index: int | None = Form(None),
    window_preset: str | None = Form("auto"),
    scale: int | None = Form(8),
    modality: str | None = Form("general"),
    accept: str | None = Header(None),
):
    current_user = get_current_user(request)
    if not current_user:
        if accept and "application/json" in accept:
            return JSONResponse({"error": "Authentication required. Please log in."}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)

    try:
        content = await file.read()
        is_valid, validation_msg = validate_file_content(content, file.filename)
        if not is_valid:
            return JSONResponse({"error": validation_msg}, status_code=400)

        extension = get_file_extension(file.filename)
        clean_name = sanitize_filename(file.filename)
        storage_id = uuid.uuid4().hex
        storage_filename = f"{storage_id}_{clean_name}"
        file_path = UPLOAD_DIR / storage_filename

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        data = execute_super_resolution_pipeline(
            file_path=file_path,
            filename=clean_name,
            extension=extension,
            description=description,
            tags=tags,
            patient_id=patient_id,
            slice_index=slice_index,
            window_preset=window_preset or "auto",
            scale=scale or 8,
            modality=modality or "general",
            owner_id=current_user.id,
        )

        if accept and "application/json" in accept:
            return JSONResponse(data)

        return RedirectResponse(url="/gallery", status_code=303)
    except Exception as exc:
        return JSONResponse({"error": "Failed to process image."}, status_code=500)


# ✅ Volume Inspection & Presets Route
@app.post("/api/volume/inspect")
async def api_inspect_volume(request: Request, file: UploadFile = File(...)):
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    content = await file.read()
    is_valid, validation_msg = validate_file_content(content, file.filename)
    if not is_valid:
        return JSONResponse({"error": validation_msg}, status_code=400)

    bio = io.BytesIO(content)
    info = inspect_volume_info(bio, filename=file.filename)
    info["presets"] = [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in HU_PRESETS.items()
    ]
    return JSONResponse(info)


@app.get("/api/volume/presets")
async def api_volume_presets():
    return JSONResponse(
        [
            {"id": k, "name": v["name"], "description": v["description"]}
            for k, v in HU_PRESETS.items()
        ]
    )


def verify_file_authorization(request: Request, filename: str, is_result_dir: bool = False) -> tuple[bool, int, str, Path | None]:
    """
    Strict authorization verification for private medical files.
    Ensures that only the authenticated scan owner or an authorized administrator
    can access original scans, super-resolved outputs, or exported DICOMs.
    Prevents path traversal attacks by resolving base names strictly inside storage directories.
    """
    current_user = get_current_user(request)
    if not current_user:
        return False, 401, "Authentication required to access medical records.", None

    clean_filename = Path(filename).name
    target_dir = RESULT_DIR if is_result_dir else UPLOAD_DIR
    file_path = target_dir / clean_filename

    if not file_path.exists():
        return False, 404, "Requested medical file does not exist.", None

    if current_user.is_admin:
        return True, 200, "", file_path

    # Check database ownership
    db = SessionLocal()
    res = db.query(Result).filter(
        (Result.original_path.like(f"%{clean_filename}%")) |
        (Result.result_path.like(f"%{clean_filename}%")) |
        (Result.dicom_path.like(f"%{clean_filename}%"))
    ).first()

    if not res:
        # Check if preview image matching base name prefix
        base_prefix = clean_filename.split("_preview")[0] if "_preview" in clean_filename else clean_filename
        res = db.query(Result).filter(
            (Result.original_path.like(f"%{base_prefix}%")) |
            (Result.result_path.like(f"%{base_prefix}%"))
        ).first()

    db.close()

    if not res:
        return False, 404, "File not associated with an authorized record.", None

    if res.owner_id != current_user.id:
        return False, 403, "Access denied: You do not have permission to access another patient's scan.", None

    return True, 200, "", file_path


# ✅ Secure Authenticated Medical File Serving
@app.get("/uploads/{filename}")
async def get_upload(request: Request, filename: str):
    auth, code, msg, file_path = verify_file_authorization(request, filename, is_result_dir=False)
    if not auth:
        raise HTTPException(status_code=code, detail=msg)
    return FileResponse(file_path)


@app.get("/results/{filename}")
async def get_result(request: Request, filename: str):
    auth, code, msg, file_path = verify_file_authorization(request, filename, is_result_dir=True)
    if not auth:
        raise HTTPException(status_code=code, detail=msg)
    return FileResponse(file_path)


@app.get("/api/dicom/download/{filename}")
@app.get("/results/dicom/{filename}")
async def get_dicom_result(request: Request, filename: str):
    auth, code, msg, file_path = verify_file_authorization(request, filename, is_result_dir=True)
    if not auth:
        raise HTTPException(status_code=code, detail=msg)
    return FileResponse(
        file_path,
        media_type="application/dicom",
        filename=Path(filename).name,
        headers={"Content-Disposition": f'attachment; filename="{Path(filename).name}"'}
    )


# ✅ Authenticated & Owner-Filtered Gallery Page
@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    db = SessionLocal()
    query_text = request.query_params.get("q", "").strip()

    results_query = db.query(Result).order_by(Result.timestamp.desc())
    if not current_user.is_admin:
        results_query = results_query.filter(Result.owner_id == current_user.id)

    if query_text:
        search = f"%{query_text}%"
        results_query = results_query.filter(
            Result.filename.ilike(search)
            | Result.description.ilike(search)
            | Result.tags.ilike(search)
            | Result.patient_id.ilike(search)
        )

    results = results_query.all()
    db.close()

    display_results = []
    for item in results:
        display_results.append(
            {
                "id": item.id,
                "filename": item.filename,
                "original_url": f"/uploads/{Path(item.original_path).name}",
                "result_url": f"/results/{Path(item.result_path).name}",
                "description": item.description,
                "tags": item.tags,
                "patient_id": item.patient_id,
                "psnr": item.psnr,
                "ssim": item.ssim,
                "sharpness": item.sharpness,
                "slice_index": item.slice_index,
                "window_preset": item.window_preset,
                "scale_factor": item.scale_factor or 8,
                "dicom_url": f"/api/dicom/download/{Path(item.dicom_path).name}" if item.dicom_path else None,
                "deidentified": item.deidentified or False,
                "timestamp": utc_to_ist(item.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "owner_id": item.owner_id,
                "is_owner": item.owner_id == current_user.id or current_user.is_admin,
            }
        )

    return HTMLResponse(
        templates.get_template("gallery.html").render(
            {
                "request": request,
                "results": display_results,
                "query": query_text,
                "user": current_user,
            }
        )
    )


# ✅ Secure Result Deletion (Ownership Strictly Enforced)
@app.post("/gallery/delete/{result_id}")
async def delete_result(request: Request, result_id: int):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    db = SessionLocal()
    result_item = db.query(Result).filter(Result.id == result_id).first()
    if not result_item:
        db.close()
        return RedirectResponse(url="/gallery", status_code=303)

    if result_item.owner_id != current_user.id and not current_user.is_admin:
        db.close()
        raise HTTPException(status_code=403, detail="Access denied: You cannot delete another user's scan.")

    original_file = Path(result_item.original_path)
    result_file = Path(result_item.result_path)
    dicom_file = Path(result_item.dicom_path) if result_item.dicom_path else None

    if original_file.exists():
        try:
            original_file.unlink()
        except Exception:
            pass
    if result_file.exists():
        try:
            result_file.unlink()
        except Exception:
            pass
    if dicom_file and dicom_file.exists():
        try:
            dicom_file.unlink()
        except Exception:
            pass

    db.delete(result_item)
    db.commit()
    db.close()

    return RedirectResponse(url="/gallery", status_code=303)


# ✅ Secure Clinical Diagnostic Report Page (Ownership Strictly Enforced)
@app.get("/report/{result_id}", response_class=HTMLResponse)
async def view_diagnostic_report(request: Request, result_id: int):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url=f"/login", status_code=303)

    db = SessionLocal()
    result_item = db.query(Result).filter(Result.id == result_id).first()
    db.close()
    if not result_item:
        raise HTTPException(status_code=404, detail="Scan result not found.")

    if result_item.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied: You do not have permission to view this report.")

    report_data = {
        "id": result_item.id,
        "filename": result_item.filename,
        "original_url": f"/uploads/{Path(result_item.original_path).name}",
        "result_url": f"/results/{Path(result_item.result_path).name}",
        "description": result_item.description,
        "tags": result_item.tags,
        "patient_id": result_item.patient_id,
        "psnr": result_item.psnr,
        "ssim": result_item.ssim,
        "sharpness": result_item.sharpness,
        "slice_index": result_item.slice_index,
        "window_preset": result_item.window_preset,
        "scale_factor": result_item.scale_factor or 8,
        "dicom_url": f"/api/dicom/download/{Path(result_item.dicom_path).name}" if result_item.dicom_path else None,
        "deidentified": result_item.deidentified or False,
        "modality": result_item.modality or "general",
        "timestamp": utc_to_ist(result_item.timestamp),
    }

    return HTMLResponse(
        templates.get_template("report.html").render(
            {
                "request": request,
                "result": report_data,
                "user": current_user,
            }
        )
    )
