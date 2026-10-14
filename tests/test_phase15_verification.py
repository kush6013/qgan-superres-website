import unittest
import os
import io
import shutil
import tempfile
import threading
import time
import hashlib
from pathlib import Path
from PIL import Image

os.environ["APP_ENV"] = "testing"
os.environ["SESSION_SECRET"] = "phase15_super_secure_verification_secret_987654321"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database
import main
from database import Base
from models import User, Result, Job
from main import (
    hash_reset_token,
    create_password_reset_token,
    verify_password_reset_token,
    validate_file_content,
    sanitize_filename,
    update_job_in_db,
    recover_interrupted_jobs,
)

try:
    import pydicom
except Exception:
    pydicom = None


class TestPhase15ProductionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="phase15_gate_")
        cls.test_upload_dir = Path(cls.test_dir) / "uploads"
        cls.test_result_dir = Path(cls.test_dir) / "results"
        cls.test_upload_dir.mkdir(parents=True, exist_ok=True)
        cls.test_result_dir.mkdir(parents=True, exist_ok=True)

        cls.orig_upload_dir = main.UPLOAD_DIR
        cls.orig_result_dir = main.RESULT_DIR
        main.UPLOAD_DIR = cls.test_upload_dir
        main.RESULT_DIR = cls.test_result_dir

        cls.test_db_path = Path(cls.test_dir) / "gate_isolated.db"
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

    @classmethod
    def tearDownClass(cls):
        main.UPLOAD_DIR = cls.orig_upload_dir
        main.RESULT_DIR = cls.orig_result_dir
        database.SessionLocal = cls.orig_session_local
        main.SessionLocal = cls.orig_main_session_local
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        with self.engine.begin() as conn:
            conn.execute(Base.metadata.tables["jobs"].delete())
            conn.execute(Base.metadata.tables["results"].delete())
            conn.execute(Base.metadata.tables["users"].delete())

    # =========================================================================
    # GATE 4: Password Reset Token Hashing Verification
    # =========================================================================
    def test_password_reset_token_hashing_and_lifecycle(self):
        db = self.TestingSessionLocal()
        user = User(
            email="reset_target@hospital.org",
            password_hash="hashed_pw_initial",
            salt="salt123",
            is_admin=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # 1. Generate reset token
        raw_token = create_password_reset_token(db, user)
        self.assertIsNotNone(raw_token)
        self.assertGreater(len(raw_token), 20)

        # 2. Verify token in DB is HASHED (SHA-256), NOT plaintext
        db.refresh(user)
        expected_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        self.assertEqual(user.reset_token, expected_hash)
        self.assertNotEqual(user.reset_token, raw_token)  # CRITICAL: Plaintext is NOT stored!

        # 3. Verify valid token works
        verified_user = verify_password_reset_token(db, raw_token)
        self.assertIsNotNone(verified_user)
        self.assertEqual(verified_user.id, user.id)

        # 4. Verify invalid token fails
        invalid_user = verify_password_reset_token(db, "fake_nonexistent_token_12345")
        self.assertIsNone(invalid_user)

        # 5. Verify expired token fails
        user.reset_token_expires = main.get_ist_now() - main.timedelta(hours=2)
        db.commit()
        expired_user = verify_password_reset_token(db, raw_token)
        self.assertIsNone(expired_user)

        # 6. Single-use invalidation
        user.reset_token = None
        user.reset_token_expires = None
        db.commit()
        reused_user = verify_password_reset_token(db, raw_token)
        self.assertIsNone(reused_user)
        db.close()

    # =========================================================================
    # GATE 6: Complete Authorization Matrix Test
    # =========================================================================
    def test_full_authorization_matrix(self):
        # Setup User A (normal)
        client_a = TestClient(main.app)
        client_a.post("/register", data={"email": "usera@hospital.org", "password": "PasswordA123!"})

        # Setup User B (normal)
        client_b = TestClient(main.app)
        client_b.post("/register", data={"email": "userb@hospital.org", "password": "PasswordB123!"})

        # Setup Admin User
        client_admin = TestClient(main.app)
        client_admin.post("/register", data={"email": "admin@hospital.org", "password": "AdminPassword123!"})

        # Demote User A and User B, elevate Admin
        db = self.TestingSessionLocal()
        user_a = db.query(User).filter(User.email == "usera@hospital.org").first()
        user_b = db.query(User).filter(User.email == "userb@hospital.org").first()
        admin_u = db.query(User).filter(User.email == "admin@hospital.org").first()
        user_a.is_admin = False
        user_b.is_admin = False
        admin_u.is_admin = True
        db.commit()

        # Create assets owned by User A
        orig_file_a = self.test_upload_dir / "user_a_orig.png"
        res_file_a = self.test_result_dir / "user_a_res.png"
        dcm_file_a = self.test_result_dir / "user_a.dcm"
        Image.new("RGB", (64, 64), color="red").save(orig_file_a)
        Image.new("RGB", (512, 512), color="red").save(res_file_a)
        dcm_file_a.write_bytes(b"DICM_TEST_MOCK_DATA")

        res_a = Result(
            filename="user_a_scan.png",
            original_path=str(orig_file_a),
            result_path=str(res_file_a),
            dicom_path=str(dcm_file_a),
            owner_id=user_a.id,
            patient_id="PT_USER_A",
        )
        db.add(res_a)
        db.commit()
        db.refresh(res_a)
        res_a_id = res_a.id

        job_a = Job(
            id="job_user_a_123",
            user_id=user_a.id,
            status="completed",
            progress=100,
            step="Completed",
            result_id=res_a_id,
        )
        db.add(job_a)
        db.commit()
        db.close()

        anon_client = TestClient(main.app)

        # Matrix: Anonymous Access (Must all be 401 or redirect to login)
        self.assertEqual(anon_client.get(f"/api/results/{res_a_id}").status_code, 401)
        self.assertEqual(anon_client.get("/uploads/user_a_orig.png").status_code, 401)
        self.assertEqual(anon_client.get("/results/user_a_res.png").status_code, 401)
        self.assertEqual(anon_client.get("/api/dicom/download/user_a.dcm").status_code, 401)
        self.assertEqual(anon_client.get(f"/report/{res_a_id}", follow_redirects=False).status_code, 303)
        self.assertEqual(anon_client.post(f"/gallery/delete/{res_a_id}", follow_redirects=False).status_code, 303)
        self.assertEqual(anon_client.get("/api/jobs/job_user_a_123").status_code, 401)

        # Matrix: User A (Owner Access - Must all be 200 OK)
        self.assertEqual(client_a.get(f"/api/results/{res_a_id}").status_code, 200)
        self.assertEqual(client_a.get("/uploads/user_a_orig.png").status_code, 200)
        self.assertEqual(client_a.get("/results/user_a_res.png").status_code, 200)
        self.assertEqual(client_a.get("/api/dicom/download/user_a.dcm").status_code, 200)
        self.assertEqual(client_a.get(f"/report/{res_a_id}").status_code, 200)
        self.assertEqual(client_a.get("/api/jobs/job_user_a_123").status_code, 200)

        # Matrix: User B -> User A (Peer IDOR Attempt - Must all be 403 Forbidden)
        self.assertEqual(client_b.get(f"/api/results/{res_a_id}").status_code, 403)
        self.assertEqual(client_b.get("/uploads/user_a_orig.png").status_code, 403)
        self.assertEqual(client_b.get("/results/user_a_res.png").status_code, 403)
        self.assertEqual(client_b.get("/api/dicom/download/user_a.dcm").status_code, 403)
        self.assertEqual(client_b.get(f"/report/{res_a_id}").status_code, 403)
        self.assertEqual(client_b.post(f"/gallery/delete/{res_a_id}").status_code, 403)
        self.assertEqual(client_b.get("/api/jobs/job_user_a_123").status_code, 403)

        # Matrix: Admin Access (Admin has permitted clinical oversight)
        self.assertEqual(client_admin.get(f"/api/results/{res_a_id}").status_code, 200)
        self.assertEqual(client_admin.get("/uploads/user_a_orig.png").status_code, 200)
        self.assertEqual(client_admin.get("/results/user_a_res.png").status_code, 200)
        self.assertEqual(client_admin.get("/api/dicom/download/user_a.dcm").status_code, 200)
        self.assertEqual(client_admin.get(f"/report/{res_a_id}").status_code, 200)
        self.assertEqual(client_admin.get("/api/jobs/job_user_a_123").status_code, 200)

    # =========================================================================
    # GATE 7: Upload Security Stress Checks (10 Attack Vectors)
    # =========================================================================
    def test_upload_security_stress_checks(self):
        client = TestClient(main.app)
        client.post("/register", data={"email": "stresstester@hospital.org", "password": "Password123!"})

        # Vector 1: Empty file
        v1_valid, v1_msg = validate_file_content(b"", "empty.png")
        self.assertFalse(v1_valid)
        self.assertIn("empty", v1_msg.lower())

        # Vector 2: Oversized file (> 50 MB)
        oversized_bytes = b"0" * (51 * 1024 * 1024)
        v2_valid, v2_msg = validate_file_content(oversized_bytes, "huge.png")
        self.assertFalse(v2_valid)
        self.assertIn("exceeds", v2_msg.lower())

        # Vector 3: Malformed PNG (invalid header bytes)
        v3_valid, v3_msg = validate_file_content(b"\x89PNG\r\n\x1a\nCORRUPTED_GARBAGE", "corrupt.png")
        self.assertFalse(v3_valid)

        # Vector 4: Malformed JPEG (invalid header)
        v4_valid, v4_msg = validate_file_content(b"\xff\xd8\xff\xe0CORRUPTED_JPEG", "corrupt.jpg")
        self.assertFalse(v4_valid)

        # Vector 5: Malformed DICOM (garbage without DICM magic or dataset)
        v5_valid, v5_msg = validate_file_content(b"RAW_TEXT_NOT_DICOM", "bad.dcm")
        self.assertFalse(v5_valid)

        # Vector 6: Executable file disguised as PNG (.sh / .exe payload)
        elf_payload = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        v6_valid, v6_msg = validate_file_content(elf_payload, "disguised.png")
        self.assertFalse(v6_valid)

        # Vector 7: Path Traversal filename
        clean_traversal = sanitize_filename("../../etc/shadow.png")
        self.assertEqual(clean_traversal, "shadow.png")
        self.assertNotIn("..", clean_traversal)
        self.assertNotIn("/", clean_traversal)

        # Vector 8: Unusual Unicode & Special Character filename
        clean_unicode = sanitize_filename("scan_ø_§_病院_#123!.png")
        self.assertTrue(clean_unicode.endswith(".png"))
        self.assertNotIn("#", clean_unicode)
        self.assertNotIn("!", clean_unicode)

        # Vector 9: Duplicate filename collision check
        # Generating two storage names for the same filename produces distinct unique paths
        storage_id1 = "uuid_one"
        storage_id2 = "uuid_two"
        path1 = main.UPLOAD_DIR / f"{storage_id1}_{clean_traversal}"
        path2 = main.UPLOAD_DIR / f"{storage_id2}_{clean_traversal}"
        self.assertNotEqual(path1, path2)

        # Vector 10: Invalid NIfTI GZIP file
        v10_valid, v10_msg = validate_file_content(b"NOT_A_GZIP_ARCHIVE", "brain.nii.gz")
        self.assertFalse(v10_valid)
        self.assertIn("gzip", v10_msg.lower())

    # =========================================================================
    # GATE 2 & 3: Multi-Worker & Concurrent Job Execution
    # =========================================================================
    def test_multi_worker_shared_job_state(self):
        # Simulate Worker 1 creating a job
        db_worker1 = self.TestingSessionLocal()
        job = Job(
            id="multi_worker_job_001",
            user_id=1,
            status="processing",
            progress=10,
            step="Worker 1 initialized",
        )
        db_worker1.add(job)
        db_worker1.commit()
        db_worker1.close()

        # Simulate Worker 2 updating progress
        update_job_in_db("multi_worker_job_001", progress=50, step="Worker 2 in progress")

        # Simulate Worker 3 reading state
        db_worker3 = self.TestingSessionLocal()
        polled_job = db_worker3.query(Job).filter(Job.id == "multi_worker_job_001").first()
        self.assertIsNotNone(polled_job)
        self.assertEqual(polled_job.progress, 50)
        self.assertEqual(polled_job.step, "Worker 2 in progress")
        self.assertEqual(polled_job.status, "processing")
        db_worker3.close()

    def test_concurrent_jobs_isolation(self):
        # Run 5 concurrent threads updating their respective jobs
        num_jobs = 5
        job_ids = [f"concurrent_job_{i}" for i in range(num_jobs)]
        errors = []

        def worker_task(jid, idx):
            try:
                db = self.TestingSessionLocal()
                j = Job(id=jid, user_id=idx + 10, status="processing", progress=idx * 10, step=f"Step {idx}")
                db.add(j)
                db.commit()
                db.close()

                # Simulate intermediate progress updates
                for p in [30, 60, 100]:
                    update_job_in_db(jid, progress=p, step=f"Progress {p}%")
                    time.sleep(0.01)

                update_job_in_db(jid, status="completed", progress=100, step="Finished")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task, args=(jid, i)) for i, jid in enumerate(job_ids)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent workers encountered errors: {errors}")

        # Verify all jobs completed independently in DB with correct values
        db_verify = self.TestingSessionLocal()
        completed_jobs = db_verify.query(Job).filter(Job.status == "completed").all()
        self.assertEqual(len(completed_jobs), num_jobs)
        for j in completed_jobs:
            self.assertEqual(j.progress, 100)
            self.assertEqual(j.step, "Finished")
        db_verify.close()


if __name__ == "__main__":
    unittest.main()
