import unittest
import os
import io
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# Use an isolated test database and test directory to protect production data
os.environ["APP_ENV"] = "testing"
os.environ["SESSION_SECRET"] = "test_super_secure_secret_key_123456789"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
import main
from database import Base
from models import User, Result, Job
from metrics import evaluate_super_resolution_quality, calculate_psnr, calculate_ssim, calculate_laplacian_sharpness
from medical_imaging import deidentify_dicom_dataset, create_dicom_from_image, inspect_volume_info

try:
    import pydicom
except Exception:
    pydicom = None


class TestSecurityAndMedicalSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create disposable temporary directory for uploads and results
        cls.test_dir = tempfile.mkdtemp(prefix="qgan_test_")
        cls.test_upload_dir = Path(cls.test_dir) / "uploads"
        cls.test_result_dir = Path(cls.test_dir) / "results"
        cls.test_upload_dir.mkdir(parents=True, exist_ok=True)
        cls.test_result_dir.mkdir(parents=True, exist_ok=True)

        # Patch directories in main module
        cls.orig_upload_dir = main.UPLOAD_DIR
        cls.orig_result_dir = main.RESULT_DIR
        main.UPLOAD_DIR = cls.test_upload_dir
        main.RESULT_DIR = cls.test_result_dir

        # Setup isolated SQLite test database
        cls.test_db_path = Path(cls.test_dir) / "test_isolated.db"
        cls.engine = create_engine(
            f"sqlite:///{cls.test_db_path}",
            connect_args={"check_same_thread": False},
        )
        cls.TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=cls.engine
        )
        Base.metadata.create_all(bind=cls.engine)

        cls.orig_session_local = database.SessionLocal
        cls.orig_main_session_local = main.SessionLocal
        database.SessionLocal = cls.TestingSessionLocal
        main.SessionLocal = cls.TestingSessionLocal

        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        # Restore original paths and database
        main.UPLOAD_DIR = cls.orig_upload_dir
        main.RESULT_DIR = cls.orig_result_dir
        database.SessionLocal = cls.orig_session_local
        main.SessionLocal = cls.orig_main_session_local
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        # Clean test tables between tests
        with self.engine.begin() as conn:
            conn.execute(Base.metadata.tables["jobs"].delete())
            conn.execute(Base.metadata.tables["results"].delete())
            conn.execute(Base.metadata.tables["users"].delete())

    # =========================================================================
    # Phase 1: Authentication Tests
    # =========================================================================
    def test_user_registration_and_login(self):
        client = TestClient(main.app)

        # Register User 1 (becomes Admin because it's first user)
        res = client.post(
            "/register",
            data={"email": "admin@hospital.org", "password": "StrongPassword123!", "full_name": "Dr. Admin"},
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)

        # Verify in DB
        db = self.TestingSessionLocal()
        admin_user = db.query(User).filter(User.email == "admin@hospital.org").first()
        self.assertIsNotNone(admin_user)
        self.assertTrue(admin_user.is_admin)
        db.close()

        # Logout
        res_logout = client.get("/logout", follow_redirects=False)
        self.assertEqual(res_logout.status_code, 303)

        # Re-login with wrong password should fail
        res_fail = client.post(
            "/login",
            data={"email": "admin@hospital.org", "password": "WrongPassword"},
        )
        self.assertIn("Invalid email or password", res_fail.text)

        # Re-login with correct password
        res_ok = client.post(
            "/login",
            data={"email": "admin@hospital.org", "password": "StrongPassword123!"},
            follow_redirects=False,
        )
        self.assertEqual(res_ok.status_code, 303)

    # =========================================================================
    # Phase 1 & 2: Authorization & IDOR Tests (User A vs User B)
    # =========================================================================
    def test_authorization_cross_user_isolation(self):
        # Register User A (admin)
        client_a = TestClient(main.app)
        client_a.post(
            "/register",
            data={"email": "userA@hospital.org", "password": "PasswordA123", "full_name": "User A"},
            follow_redirects=True,
        )

        # Register User B (normal user)
        client_b = TestClient(main.app)
        client_b.post(
            "/register",
            data={"email": "userB@hospital.org", "password": "PasswordB123", "full_name": "User B"},
            follow_redirects=True,
        )

        db = self.TestingSessionLocal()
        user_a = db.query(User).filter(User.email == "userA@hospital.org").first()
        user_b = db.query(User).filter(User.email == "userB@hospital.org").first()
        # Set User A as normal user for testing peer isolation
        user_a.is_admin = False
        user_b.is_admin = False
        db.commit()

        # Create a mock file and Result owned by User A
        test_img_path = self.test_upload_dir / "user_a_orig.png"
        test_res_path = self.test_result_dir / "user_a_sr.png"
        Image.new("RGB", (64, 64), color="blue").save(test_img_path)
        Image.new("RGB", (512, 512), color="blue").save(test_res_path)

        res_a = Result(
            filename="user_a_scan.png",
            original_path=str(test_img_path),
            result_path=str(test_res_path),
            owner_id=user_a.id,
            patient_id="PATIENT_SECRET_999",
            scale_factor=8,
        )
        db.add(res_a)
        db.commit()
        db.refresh(res_a)
        res_a_id = res_a.id
        db.close()

        # Anonymous client cannot view report
        anon_client = TestClient(main.app)
        res_anon_report = anon_client.get(f"/report/{res_a_id}", follow_redirects=False)
        self.assertEqual(res_anon_report.status_code, 303)  # redirected to login

        # Anonymous client cannot download User A's file
        res_anon_file = anon_client.get("/uploads/user_a_orig.png")
        self.assertEqual(res_anon_file.status_code, 401)

        # User B CANNOT view User A's result in API (IDOR attempt)
        res_b_api = client_b.get(f"/api/results/{res_a_id}")
        self.assertEqual(res_b_api.status_code, 403)

        # User B CANNOT download User A's uploaded file
        res_b_file = client_b.get("/uploads/user_a_orig.png")
        self.assertEqual(res_b_file.status_code, 403)

        # User B CANNOT view User A's diagnostic report
        res_b_report = client_b.get(f"/report/{res_a_id}")
        self.assertEqual(res_b_report.status_code, 403)

        # User B CANNOT delete User A's result
        res_b_del = client_b.post(f"/gallery/delete/{res_a_id}")
        self.assertEqual(res_b_del.status_code, 403)

        # User A CAN access their own report
        res_a_report = client_a.get(f"/report/{res_a_id}")
        self.assertEqual(res_a_report.status_code, 200)

        # User A CAN access their own file
        res_a_file = client_a.get("/uploads/user_a_orig.png")
        self.assertEqual(res_a_file.status_code, 200)

    # =========================================================================
    # Phase 1: Upload Validation & Path Traversal Prevention
    # =========================================================================
    def test_upload_validation_and_path_traversal(self):
        client = TestClient(main.app)
        client.post(
            "/register",
            data={"email": "uploader@hospital.org", "password": "Password123"},
            follow_redirects=True,
        )

        # 1. Reject empty file
        empty_res = client.post(
            "/upload/async",
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(empty_res.status_code, 400)

        # 2. Reject unsupported extension (.sh / .py)
        bad_ext_res = client.post(
            "/upload/async",
            files={"file": ("malicious.sh", b"#!/bin/bash\necho hack", "application/x-sh")},
        )
        self.assertEqual(bad_ext_res.status_code, 400)

        # 3. Reject corrupted image masquerading as PNG
        fake_png = client.post(
            "/upload/async",
            files={"file": ("fake.png", b"NOT_A_REAL_PNG_HEADER", "image/png")},
        )
        self.assertEqual(fake_png.status_code, 400)

        # 4. Valid PNG upload with path traversal in filename
        img_buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color="red").save(img_buffer, format="PNG")
        png_bytes = img_buffer.getvalue()

        # Filename attempts to escape: "../../etc/evil.png"
        traversal_res = client.post(
            "/upload/async",
            files={"file": ("../../evil.png", png_bytes, "image/png")},
        )
        self.assertEqual(traversal_res.status_code, 200)
        data = traversal_res.json()
        self.assertIn("job_id", data)

        # Confirm no files were written outside the upload directory
        evil_outside = Path("evil.png")
        self.assertFalse(evil_outside.exists())

    # =========================================================================
    # Phase 2: Persistent Background Job Status
    # =========================================================================
    def test_persistent_background_job_tracking(self):
        client = TestClient(main.app)
        client.post(
            "/register",
            data={"email": "jobuser@hospital.org", "password": "Password123"},
            follow_redirects=True,
        )

        db = self.TestingSessionLocal()
        user = db.query(User).filter(User.email == "jobuser@hospital.org").first()

        # Insert a job into the persistent database
        test_job = Job(
            id="job-uuid-12345",
            user_id=user.id,
            status="processing",
            progress=45,
            step="Extracting radiologic slice...",
        )
        db.add(test_job)
        db.commit()
        db.close()

        # Verify job is queryable via API
        res = client.get("/api/jobs/job-uuid-12345")
        self.assertEqual(res.status_code, 200)
        job_data = res.json()
        self.assertEqual(job_data["job_id"], "job-uuid-12345")
        self.assertEqual(job_data["status"], "processing")
        self.assertEqual(job_data["progress"], 45)

    def test_interrupted_job_recovery_on_restart(self):
        db = self.TestingSessionLocal()
        # Simulate a stalled job left in 'processing' state from a previous crash/restart
        stalled_job = Job(
            id="stalled-job-999",
            status="processing",
            progress=50,
            step="Running inference...",
        )
        db.add(stalled_job)
        db.commit()
        db.close()

        # Trigger recovery function
        main.recover_interrupted_jobs()

        # Verify job was cleanly transitioned to error status
        db = self.TestingSessionLocal()
        recovered = db.query(Job).filter(Job.id == "stalled-job-999").first()
        self.assertEqual(recovered.status, "error")
        self.assertIn("interrupted", recovered.step.lower())
        db.close()

    # =========================================================================
    # Phase 3: Metrics Verification & Reference Labeling
    # =========================================================================
    def test_image_quality_metrics_labeling(self):
        # 1. Normal image evaluation against bicubic baseline
        orig = Image.new("RGB", (32, 32), color=(100, 100, 100))
        enhanced = Image.new("RGB", (256, 256), color=(110, 110, 110))

        metrics = evaluate_super_resolution_quality(orig, enhanced)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("sharpness", metrics)
        self.assertEqual(metrics["reference_type"], "bicubic_baseline")
        self.assertIn("algorithmic", metrics["clinical_disclaimer"])

        # 2. Evaluation against provided ground truth
        gt = Image.new("RGB", (256, 256), color=(110, 110, 110))
        gt_metrics = evaluate_super_resolution_quality(orig, enhanced, ground_truth_img=gt)
        self.assertEqual(gt_metrics["reference_type"], "ground_truth")
        self.assertIn("ground truth", gt_metrics["reference_description"])

        # 3. Constant/empty image edge case
        zero_img = Image.new("RGB", (32, 32), color=0)
        zero_enh = Image.new("RGB", (256, 256), color=0)
        zero_metrics = evaluate_super_resolution_quality(zero_img, zero_enh)
        self.assertEqual(zero_metrics["sharpness"], 0.0)

    # =========================================================================
    # Phase 4: DICOM PHI De-identification & Export
    # =========================================================================
    @unittest.skipIf(pydicom is None, "pydicom not installed")
    def test_dicom_phi_deidentification_and_export(self):
        # Create minimal test DICOM dataset
        ds = pydicom.dataset.Dataset()
        ds.PatientName = "DOE^JOHN^A"
        ds.PatientID = "SECRET_SSN_12345"
        ds.PatientBirthDate = "19800101"
        ds.InstitutionName = "Metropolitan Hospital"
        ds.ReferringPhysicianName = "Dr. Specialist"

        # Apply HIPAA de-identification
        modified = deidentify_dicom_dataset(ds, custom_patient_id="ANON_CLINICAL_01")
        self.assertTrue(modified)
        self.assertEqual(ds.PatientName, "ANONYMIZED^PATIENT")
        self.assertEqual(ds.PatientID, "ANON_CLINICAL_01")
        self.assertEqual(ds.PatientBirthDate, "")
        self.assertEqual(ds.InstitutionName, "ANONYMOUS_HOSPITAL")
        self.assertEqual(ds.PatientIdentityRemoved, "YES")

        # Test DICOM export from image
        test_img = Image.new("L", (128, 128), color=128)
        dcm_out = self.test_result_dir / "test_export.dcm"
        saved = create_dicom_from_image(test_img, dcm_out, patient_id="EXPORT_TEST_01")
        self.assertIsNotNone(saved)
        self.assertTrue(dcm_out.exists())

        # Verify exported file is valid and readable by pydicom
        read_back = pydicom.dcmread(str(dcm_out))
        self.assertEqual(read_back.Rows, 128)
        self.assertEqual(read_back.Columns, 128)
        self.assertEqual(read_back.PatientID, "EXPORT_TEST_01")


if __name__ == "__main__":
    unittest.main()
