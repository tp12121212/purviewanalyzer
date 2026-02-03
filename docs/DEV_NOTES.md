# Development Notes

## File ingestion manual test checklist

- Office documents (.docx/.pptx/.xlsx): upload, verify text appears in Input.
- Legacy Office (.doc/.ppt/.xls): upload, verify extraction or friendly error if dependencies are missing.
- PDF (text-based): upload, verify text extracted.
- PDF (scanned/images): upload, verify OCR text extracted.
- Email (.eml/.msg) with attachments: upload, verify subject/body + attachment text included.
- Images (.jpg/.png/.tif/.gif): upload, verify OCR text extracted.
- Archives (.zip/.7z/.rar) with nested files: upload, verify recursive extraction, safe limits enforced.

Notes:
- Large archives are capped at ~200MB total extracted size and 2000 files.
- Some legacy formats require optional system dependencies (e.g., textract helpers).
- Optional OCR engine: set `USE_EASYOCR=1` to use EasyOCR instead of Tesseract.
