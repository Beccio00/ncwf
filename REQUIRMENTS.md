# NWCF - REQUIRMENTS

## 1. Overview

NWCF is a project designed to simplify the house managers' process concerned to insert the daily turists datas in the applications of the region Turismo5 and ... .
The user can add only the barcode of the digital card and the apllication autmatically privide as output a . ...  file supported by Turismo5 and ... .

---------------

## 2. Functional Rquirements

1. **Input Data**
   - Upload one or more images containing identity documents (ID cards or passports).
   - Support for common formats (JPG, PNG).

2. **Information Extraction**
   - Detect and decode barcodes (e.g., PDF417).
   - Parse the decoded data into a structured format (first name, last name, date of birth, gender, nationality, document number, issue/expiry dates, etc.).

3. **File Generation**
   - Create export files in the required formats:
        .....

   - Each file must comply with the official ministerial or regional specifications.

4. **User Interface**
   - Provide both a command-line interface and a simple GUI/web interface.
   - Allow users to:
     - Upload documents.
     - Review and confirm extracted data.
     - Export the final file.

---

## Non-Functional Requirements

- **Portability**: the software should run on common operating systems (Windows, Linux, macOS).
- **Accuracy**: reliable decoding of barcodes, even from imperfect images (with pre-processing support).
- **Scalability**: ability to process multiple documents in batch mode.
- **Privacy & Security**:
  - Safe handling of personal data (GDPR compliance).
  - Default local-only processing, with no automatic upload to external servers.
  - Secure deletion or encryption of temporary files.
- **Modularity**: the system should be organized into clear modules (decoding, parsing, file generation, UI).

---