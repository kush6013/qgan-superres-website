from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None


# Standard Clinical Radiologic Hounsfield Unit (HU) Presets
HU_PRESETS: Dict[str, Dict[str, Any]] = {
    "auto": {
        "name": "Auto Contrast",
        "description": "Full dynamic range min-max scaling",
        "center": None,
        "width": None,
    },
    "soft_tissue": {
        "name": "Soft Tissue",
        "description": "Optimized for abdominal organs, muscles, and glands (C: 40, W: 400)",
        "center": 40,
        "width": 400,
    },
    "bone": {
        "name": "Bone Window",
        "description": "Accentuates cortex, trabecular structures, and fractures (C: 400, W: 1800)",
        "center": 400,
        "width": 1800,
    },
    "lung": {
        "name": "Lung Window",
        "description": "Accentuates pulmonary parenchyma and fine bronchial air (C: -600, W: 1500)",
        "center": -600,
        "width": 1500,
    },
    "brain": {
        "name": "Brain Window",
        "description": "Sharp differentiation between gray and white matter (C: 40, W: 80)",
        "center": 40,
        "width": 80,
    },
}


def apply_hu_window(
    array: np.ndarray,
    center: Optional[float] = None,
    width: Optional[float] = None
) -> np.ndarray:
    """
    Applies standard clinical Hounsfield Unit (HU) window leveling.
    Formula:
      lower = center - width / 2
      upper = center + width / 2
      normalized = (clip(array, lower, upper) - lower) / width * 255
    """
    array = np.nan_to_num(array.astype(np.float32))

    if center is None or width is None or width <= 0:
        # Fall back to robust min-max normalization
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val <= min_val:
            return np.zeros_like(array, dtype=np.uint8)
        norm = (array - min_val) / (max_val - min_val)
        return (norm * 255.0).clip(0, 255).astype(np.uint8)

    lower = center - (width / 2.0)
    upper = center + (width / 2.0)

    clipped = np.clip(array, lower, upper)
    normalized = (clipped - lower) / float(width)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def get_volume_data(file_input: Any, filename: str = "") -> Tuple[np.ndarray, bool, int]:
    """
    Loads raw 2D or 3D numerical volume array from DICOM, NIfTI, or standard image.
    Supports on-disk paths or in-memory BytesIO streams without creating temporary files.
    Returns: (array, is_3d, total_slices)
    """
    if isinstance(file_input, Path):
        suffix = file_input.name.lower()
        target = str(file_input)
    elif hasattr(file_input, "name") and file_input.name:
        suffix = str(file_input.name).lower()
        target = file_input
    else:
        suffix = (filename or "").lower()
        target = file_input

    if suffix.endswith(".nii.gz") or suffix.endswith(".nii"):
        if nib is None:
            raise RuntimeError("NIfTI support requires the 'nibabel' package.")
        if isinstance(target, str):
            nii = nib.load(target)
        else:
            file_holder = nib.FileHolder(fileobj=target)
            nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
        data = np.asarray(nii.get_fdata(), dtype=np.float32)

        # Ensure spatial slices are ordered along axis 0
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
            return data, True, int(data.shape[0])
        elif data.ndim == 4:
            mid_t = data.shape[-1] // 2
            data3d = np.transpose(data[..., mid_t], (2, 1, 0))
            return data3d, True, int(data3d.shape[0])
        return data, False, 1

    if suffix.endswith(".dcm"):
        if pydicom is None:
            raise RuntimeError("DICOM support requires the 'pydicom' package.")
        ds = pydicom.dcmread(target, force=True)
        raw = ds.pixel_array.astype(np.float32)

        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        hu_data = raw * float(slope) + float(intercept)

        if hu_data.ndim >= 3:
            return hu_data, True, int(hu_data.shape[0])
        return hu_data, False, 1

    # Standard 2D image
    img = Image.open(target).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr, False, 1


def inspect_volume_info(file_input: Any, filename: str = "") -> Dict[str, Any]:
    """
    Inspects whether an upload is a volumetric scan directly in memory without temp files.
    """
    try:
        data, is_3d, total_slices = get_volume_data(file_input, filename=filename)
        default_slice = (total_slices // 2) if is_3d else 0
        return {
            "is_volume": is_3d,
            "total_slices": total_slices,
            "default_slice": default_slice,
            "shape": list(data.shape),
        }
    except Exception as e:
        return {
            "is_volume": False,
            "total_slices": 1,
            "default_slice": 0,
            "error": str(e),
            "presets": list(HU_PRESETS.keys()),
        }


def render_slice_image(
    file_path: Path,
    slice_index: Optional[int] = None,
    window_preset: str = "auto"
) -> Image.Image:
    """
    Extracts a specific slice with a given Hounsfield Unit windowing preset,
    returning a normalized PIL Image in RGB format.
    """
    data, is_3d, total_slices = get_volume_data(file_path)

    if is_3d and total_slices > 1:
        if slice_index is None:
            slice_index = total_slices // 2
        else:
            slice_index = max(0, min(total_slices - 1, int(slice_index)))
        slice_2d = data[slice_index]
    else:
        slice_2d = data

    if slice_2d.ndim >= 3:
        # If still multidimensional, take center
        slice_2d = np.take(slice_2d, slice_2d.shape[0] // 2, axis=0)

    preset = HU_PRESETS.get(window_preset.lower(), HU_PRESETS["auto"])
    windowed_uint8 = apply_hu_window(
        slice_2d,
        center=preset.get("center"),
        width=preset.get("width")
    )

    return Image.fromarray(windowed_uint8).convert("RGB")


# HIPAA Safe Harbor tags that contain Protected Health Information (PHI)
HIPAA_PHI_TAGS = [
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "PhysiciansOfRecord",
    "PerformingPhysicianName",
    "OperatorName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "AccessionNumber",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientAge",
    "PatientComments",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
]


def deidentify_dicom_dataset(ds: Any, custom_patient_id: Optional[str] = None) -> bool:
    """
    De-identifies a pydicom Dataset according to HIPAA Safe Harbor guidelines.
    Removes identifiable patient and physician names, dates of birth, and institutional tags.
    Strips vendor-specific private tags.

    LIMITATION WARNING:
    Tag-based de-identification strips header metadata. It does NOT automatically
    redact burned-in pixel text annotations (common in ultrasound or fluoroscopy).
    Visual inspection is required for complete de-identification.
    """
    if pydicom is None or ds is None:
        return False

    modified = False

    # Remove vendor private tags if supported
    if hasattr(ds, "remove_private_tags"):
        try:
            ds.remove_private_tags()
            modified = True
        except Exception:
            pass

    for tag_name in HIPAA_PHI_TAGS:
        if hasattr(ds, tag_name):
            if tag_name == "PatientName":
                setattr(ds, tag_name, "ANONYMIZED^PATIENT")
            elif tag_name == "InstitutionName":
                setattr(ds, tag_name, "ANONYMOUS_HOSPITAL")
            else:
                setattr(ds, tag_name, "")
            modified = True

    # Patient ID handling
    if custom_patient_id:
        ds.PatientID = str(custom_patient_id)
        modified = True
    else:
        ds.PatientID = "ANONYMIZED_ID"
        modified = True

    if modified:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "HIPAA Safe Harbor Tag Sanitization"

    return modified


def deidentify_dicom_file(
    file_path: Path,
    output_path: Optional[Path] = None,
    custom_patient_id: Optional[str] = None
) -> Tuple[Path, bool]:
    """
    De-identifies an on-disk DICOM file, stripping HIPAA Protected Health Information.
    """
    if pydicom is None:
        return file_path, False

    out = output_path or file_path
    try:
        ds = pydicom.dcmread(str(file_path), force=True)
        was_deidentified = deidentify_dicom_dataset(ds, custom_patient_id=custom_patient_id)
        ds.save_as(str(out), write_like_original=False)
        return out, was_deidentified
    except Exception:
        return file_path, False


def create_dicom_from_image(
    image: Image.Image,
    output_path: Path,
    source_dicom_path: Optional[Path] = None,
    patient_id: Optional[str] = None,
    series_description: str = "QGAN Super-Resolution Enhanced"
) -> Optional[Path]:
    """
    Exports an enhanced PIL image as a clinically compliant DICOM (.dcm) file.
    If source_dicom_path is available, preserves original study and acquisition metadata while updating pixels and dimensions.
    Otherwise, creates a compliant Secondary Capture (SC) DICOM object.
    """
    if pydicom is None:
        return None

    try:
        gray_img = image.convert("L")
        arr = np.asarray(gray_img, dtype=np.uint8)
        rows, cols = arr.shape

        if source_dicom_path and source_dicom_path.exists():
            ds = pydicom.dcmread(str(source_dicom_path), force=True)
            deidentify_dicom_dataset(ds, custom_patient_id=patient_id)

            ds.Rows = rows
            ds.Columns = cols
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = arr.tobytes()

            # Update identifiers to mark as reconstructed series
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesDescription = series_description
            ds.save_as(str(output_path))
            return output_path
        else:
            # Generate minimal Secondary Capture IOD DICOM
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = pydicom.dataset.FileDataset(
                str(output_path),
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128
            )

            ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.Modality = "OT"
            ds.PatientName = "ANONYMIZED^PATIENT"
            ds.PatientID = str(patient_id or "ANON_ID")
            ds.SeriesDescription = series_description

            ds.Rows = rows
            ds.Columns = cols
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = arr.tobytes()

            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.save_as(str(output_path))
            return output_path
    except Exception as exc:
        print(f"Warning: Failed to create DICOM export: {exc}")
        return None

