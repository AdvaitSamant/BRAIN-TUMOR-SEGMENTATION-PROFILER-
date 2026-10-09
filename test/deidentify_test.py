"""Shows a DICOM header before/after sanitization and fails loudly if any PHI survives."""
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from deidentify import snapshot, sanitized_copy

meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
meta.MediaStorageSOPClassUID = MRImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
ds = FileDataset("synthetic.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
ds.PatientName, ds.PatientID, ds.PatientBirthDate = "DOE^JOHN", "MRN-483920", "19710314"
ds.InstitutionName, ds.ReferringPhysicianName = "City General Hospital", "SMITH^ALICE^DR"
ds.StudyDate, ds.AccessionNumber, ds.StudyDescription = "20250612", "ACC99120", "MRI BRAIN - J DOE"
ds.add_new(0x00110010, "LO", "VENDOR PRIVATE"); ds.add_new(0x00111001, "LO", "private payload")
ds.Rows = ds.Columns = 4; ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PixelRepresentation = 1, "MONOCHROME2", 0
ds.PixelData = np.zeros((4, 4), np.uint16).tobytes()

before, clean = snapshot(ds), sanitized_copy(ds)
shown, after = snapshot(clean), snapshot(clean, include_uids=False)
print(f"{'TAG':32}{'BEFORE':30}AFTER")
for k, v in before.items():
    print(f"{k:32}{v[:28]:30}{shown.get(k, '<removed>')[:40]}")
print("\nPatientIdentityRemoved:", clean.PatientIdentityRemoved)
assert not after, f"PHI still present: {after}"
print("PASS: no identifying header fields remain")
