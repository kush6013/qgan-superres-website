from fastapi import FastAPI, Request, UploadFile, File, Header, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import shutil
from datetime import datetime, timedelta, timezone
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import hashlib
import secrets

from database import SessionLocal, init_db
from models import Result, User
from models.medical_classifier import MedicalImageClassifier
from models.qgan_model import QGANGenerator, load_generator as load_qgan_generator

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency
    pydicom = None

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

# ✅ App init
app = FastAPI(title="QGAN Medical Image Super-Resolution")
app.add_middleware(SessionMiddleware, secret_key="change_this_secret_for_production")

# ✅ Directories
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPSCALE_FACTOR = 8
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
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

# ✅ Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

templates = Jinja2Templates(directory="/home/pinkee-sharma/qgan-superres-website/templates")
templates.env.cache = None

# ✅ Database setup
init_db()

# Authentication helpers

SESSION_USER_KEY = "user_id"


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    return hash_password(password, salt) == password_hash


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


def create_password_reset_token(db, user: User) -> str:
    token = secrets.token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expires = get_ist_now() + timedelta(hours=1)
    db.add(user)
    db.commit()
    return token


def verify_password_reset_token(db, token: str) -> User | None:
    user = db.query(User).filter(User.reset_token == token).first()
    if not user or not user.reset_token_expires:
        return None
    if user.reset_token_expires < get_ist_now():
        return None
    return user


# QGAN model integration
MODEL_PATH = Path("models/qgan_generator.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_generator: QGANGenerator | None = None


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

CLASSIFIER_PATH = Path("models/medical_classifier.pt")
medical_classifier = MedicalImageClassifier(DEVICE, checkpoint_path=CLASSIFIER_PATH)

def utc_to_ist(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to Indian Standard Time (IST, UTC+5:30)"""
    ist_offset = timedelta(hours=5, minutes=30)
    return utc_dt + ist_offset

def get_ist_now() -> datetime:
    """Get current time in Indian Standard Time (IST, UTC+5:30)"""
    utc_now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    return utc_now + ist_offset


def get_file_extension(filename: str) -> str:
    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower_name).suffix


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


def load_pil_image(file_path: Path, extension: str | None = None) -> Image.Image:
    extension = extension or get_file_extension(file_path.name)

    if extension in IMAGE_EXTENSIONS:
        return Image.open(file_path).convert("RGB")

    if extension == ".dcm":
        if pydicom is None:
            raise RuntimeError("DICOM support requires the 'pydicom' package.")
        dataset = pydicom.dcmread(str(file_path))
        pixel_array = normalize_image_array(dataset.pixel_array)
        return Image.fromarray(pixel_array).convert("RGB")

    if extension == ".nii.gz" or extension == ".nii":
        if nib is None:
            raise RuntimeError("NIfTI support requires the 'nibabel' package.")
        volume = nib.load(str(file_path)).get_fdata()
        pixel_array = normalize_image_array(volume)
        return Image.fromarray(pixel_array).convert("RGB")

    raise ValueError(f"Unsupported file extension: {extension}")


def ensure_preview_image(file_path: Path, extension: str) -> Path:
    if extension in IMAGE_EXTENSIONS:
        return file_path

    preview_path = get_preview_path(file_path, extension)
    preview_image = load_pil_image(file_path, extension)
    preview_image.save(preview_path)
    return preview_path


def super_resolve_image(low_res_path: str) -> str:
    source_path = Path(low_res_path)
    img = load_pil_image(source_path, get_file_extension(source_path.name))
    model = load_generator()

    if model is None:
        high_res = img.resize((img.width * UPSCALE_FACTOR, img.height * UPSCALE_FACTOR), Image.BICUBIC)
    else:
        input_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            output_tensor = model(input_tensor)

        if output_tensor.min() < 0 or output_tensor.max() > 1:
            output_tensor = (output_tensor + 1.0) / 2.0

        output_tensor = output_tensor.clamp(0, 1).cpu().squeeze(0)
        high_res = TF.to_pil_image(output_tensor)

    result_filename = f"sr_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.png"
    result_path = RESULT_DIR / result_filename
    high_res.save(result_path)
    return str(result_path)


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
                "original_url": f"/uploads/{Path(item.original_path).name}",
                "result_url": f"/results/{Path(item.result_path).name}",
                "timestamp": item.timestamp.isoformat(),
            }
            for item in results
        ]
    )


@app.get("/api/results/{result_id}")
async def api_get_result(result_id: int):
    db = SessionLocal()
    item = db.query(Result).filter(Result.id == result_id).first()
    db.close()
    if not item:
        return JSONResponse({"error": "Result not found."}, status_code=404)
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


# ✅ Upload Route
@app.post("/upload/")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),
    patient_id: str | None = Form(None),
    accept: str | None = Header(None)
):
    try:
        extension = get_file_extension(file.filename)
        if extension not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                {
                    "error": "Invalid file type. Please upload a PNG, JPG, JPEG, DICOM, or NIfTI medical image."
                },
                status_code=400,
            )

        if extension in IMAGE_EXTENSIONS and file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            return JSONResponse(
                {
                    "error": "Invalid image content type. Please upload a PNG, JPG, JPEG, DICOM, or NIfTI medical image."
                },
                status_code=400,
            )

        file_path = UPLOAD_DIR / file.filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        preview_path = ensure_preview_image(file_path, extension)

        # Detect whether the upload looks medical, but do not hard-reject supported formats.
        detected_as_medical = is_medical_image(preview_path)

        # Process image
        result_path = super_resolve_image(str(preview_path))

        current_user = get_current_user(request)
        owner_id = current_user.id if current_user else None

        # Save to DB
        db = SessionLocal()
        result = Result(
            filename=file.filename,
            original_path=str(preview_path),
            result_path=result_path,
            description=description,
            tags=tags,
            patient_id=patient_id,
            owner_id=owner_id,
        )
        db.add(result)
        db.commit()
        db.close()

        original_url = f"/uploads/{preview_path.name}"
        result_url = f"/results/{Path(result_path).name}"

        if accept and "application/json" in accept:
            message = "QGAN enhancement finished successfully."
            if not detected_as_medical:
                message += " Format accepted, but the medical-image classifier could not confidently verify this upload as a scan."
            return JSONResponse(
                {
                    "original": original_url,
                    "result": result_url,
                    "message": message,
                    "detected_as_medical": detected_as_medical,
                }
            )

        return RedirectResponse(url="/gallery", status_code=303)
    except Exception as exc:
        error_message = str(exc)
        return JSONResponse({"error": error_message}, status_code=500)

# ✅ Serve Uploaded Images
@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    file = UPLOAD_DIR / filename
    if file.exists():
        return FileResponse(file)
    return {"error": "File not found"}

# ✅ Serve Result Images
@app.get("/results/{filename}")
async def get_result(filename: str):
    file = RESULT_DIR / filename
    if file.exists():
        return FileResponse(file)
    return {"error": "File not found"}

# ✅ Gallery Page
@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    db = SessionLocal()
    current_user = get_current_user(request)
    query_text = request.query_params.get("q", "").strip()

    results_query = db.query(Result).order_by(Result.timestamp.desc())
    if current_user:
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
                "timestamp": utc_to_ist(item.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "owner_id": item.owner_id,
                "is_owner": current_user is not None and item.owner_id is not None and item.owner_id == current_user.id,
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


@app.post("/gallery/delete/{result_id}")
async def delete_result(request: Request, result_id: int):
    db = SessionLocal()
    result_item = db.query(Result).filter(Result.id == result_id).first()
    if not result_item:
        db.close()
        return RedirectResponse(url="/gallery", status_code=303)

    original_file = Path(result_item.original_path)
    result_file = Path(result_item.result_path)
    if original_file.exists():
        original_file.unlink()
    if result_file.exists():
        result_file.unlink()

    db.delete(result_item)
    db.commit()
    db.close()

    return RedirectResponse(url="/gallery", status_code=303)
