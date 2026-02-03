from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader
import fitz  # pymupdf

from email import policy
from email.parser import BytesParser

_OCR_CACHE: dict = {}


@dataclass
class ExtractionResult:
    filename: str
    ok: bool
    message: str


SUPPORTED_EXTENSIONS = {
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".rtf",
    ".odt",
    ".pdf",
    ".eml",
    ".msg",
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".gif",
    ".zip",
    ".7z",
    ".rar",
}


def _safe_join(base: Path, *paths: str) -> Path:
    candidate = (base.joinpath(*paths)).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise ValueError("Path traversal detected")
    return candidate


def _limit_extraction(root: Path, max_files: int, max_bytes: int) -> None:
    total = 0
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            if count > max_files:
                raise ValueError("Too many files extracted")
            total += path.stat().st_size
            if total > max_bytes:
                raise ValueError("Extracted content too large")


def _extract_zip(archive_path: Path, dest: Path, max_files: int, max_bytes: int) -> None:
    import zipfile

    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ValueError("Too many files in zip")
        total = sum(i.file_size for i in infos)
        if total > max_bytes:
            raise ValueError("Zip content too large")
        for info in infos:
            if info.is_dir():
                continue
            target = _safe_join(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _extract_7z(archive_path: Path, dest: Path, max_files: int, max_bytes: int) -> None:
    import py7zr

    with py7zr.SevenZipFile(archive_path, mode="r") as zf:
        names = zf.getnames()
        if len(names) > max_files:
            raise ValueError("Too many files in 7z")
        zf.extractall(path=dest)
    _limit_extraction(dest, max_files, max_bytes)


def _extract_rar(archive_path: Path, dest: Path, max_files: int, max_bytes: int) -> None:
    import rarfile

    with rarfile.RarFile(archive_path) as rf:
        infos = rf.infolist()
        if len(infos) > max_files:
            raise ValueError("Too many files in rar")
        total = sum(i.file_size for i in infos)
        if total > max_bytes:
            raise ValueError("Rar content too large")
        for info in infos:
            if info.is_dir():
                continue
            target = _safe_join(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with rf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _prep_variants(image: Image.Image) -> list[Image.Image]:
    gray = ImageOps.grayscale(image)
    scale = 2
    gray = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    auto = ImageOps.autocontrast(gray)
    bw_160 = auto.point(lambda x: 0 if x < 160 else 255, mode="1")
    bw_190 = auto.point(lambda x: 0 if x < 190 else 255, mode="1")
    return [gray, auto, bw_160, bw_190]


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    stripped = text.strip()
    if not stripped:
        return 0.0
    letters = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    spaces = sum(ch.isspace() for ch in stripped)
    total = len(stripped)
    return (letters + digits) / max(total - spaces, 1)


def _ocr_image(image: Image.Image) -> str:
    if os.getenv("USE_EASYOCR", "0").lower() in {"1", "true", "yes", "on"}:
        try:
            import easyocr
            import numpy as np

            if "_easyocr_reader" not in _OCR_CACHE:
                _OCR_CACHE["_easyocr_reader"] = easyocr.Reader(["en"], gpu=False)
            reader = _OCR_CACHE["_easyocr_reader"]
            arr = np.array(image.convert("RGB"))
            results = reader.readtext(arr, detail=0, paragraph=True)
            return "\n".join([r for r in results if r]).strip()
        except Exception:
            pass

    lang = os.getenv("OCR_LANG", "eng")
    configs = [
        os.getenv("OCR_CONFIG", "--oem 1 --psm 6"),
        "--oem 1 --psm 4",
    ]
    best_text = ""
    best_quality = 0.0
    for variant in _prep_variants(image):
        for cfg in configs:
            text = pytesseract.image_to_string(variant, lang=lang, config=cfg)
            quality = _text_quality(text)
            if quality > best_quality:
                best_quality = quality
                best_text = text

    # MRZ pass (bottom 35%) using OCRB if available
    mrz_text = ""
    try:
        ocrb_config = os.getenv(
            "OCR_MRZ_CONFIG",
            "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
        )
        mrz_lang = os.getenv("OCR_MRZ_LANG", "ocrb")
        bottom = image.crop((0, int(image.height * 0.65), image.width, image.height))
        mrz_variant = _prep_variants(bottom)[2]
        mrz_text = pytesseract.image_to_string(mrz_variant, lang=mrz_lang, config=ocrb_config)
    except Exception:
        pass

    return "\n".join([t for t in [best_text.strip(), mrz_text.strip()] if t]).strip()


def _extract_pdf(path: Path) -> str:
    text_parts: List[str] = []
    reader = PdfReader(str(path))
    has_text = False
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            has_text = True
        text_parts.append(page_text)
    combined = "\n".join(text_parts).strip()
    if has_text and combined:
        return combined

    # OCR fallback for scanned PDFs (try PyMuPDF OCR text first)
    doc = fitz.open(str(path))
    ocr_text_parts: List[str] = []
    for page in doc:
        try:
            textpage = page.get_textpage_ocr(language=os.getenv("OCR_LANG", "eng"))
            page_text = textpage.extractText().strip()
            if page_text and _text_quality(page_text) > 0.5:
                ocr_text_parts.append(page_text)
                continue
        except Exception:
            pass

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text_parts.append(_ocr_image(img))

    if ocr_text_parts:
        return "\n".join([p for p in ocr_text_parts if p]).strip()

    # OCR fallback for scanned PDFs
    ocr_parts: List[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_parts.append(_ocr_image(img))
    return "\n".join(ocr_parts).strip()


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _extract_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: List[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                if shape.text:
                    parts.append(shape.text)
    return "\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts: List[str] = []
    for sheet in wb.worksheets:
        parts.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            values = [str(cell) for cell in row if cell is not None]
            if values:
                parts.append("\t".join(values))
    return "\n".join(parts)


def _extract_xls(path: Path) -> str:
    import xlrd

    wb = xlrd.open_workbook(str(path))
    parts: List[str] = []
    for sheet in wb.sheets():
        parts.append(f"# Sheet: {sheet.name}")
        for row_idx in range(sheet.nrows):
            row = sheet.row_values(row_idx)
            values = [str(cell) for cell in row if cell not in ("", None)]
            if values:
                parts.append("\t".join(values))
    return "\n".join(parts)


def _extract_rtf(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    return rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))


def _extract_odt(path: Path) -> str:
    import odf.text
    import odf.opendocument

    doc = odf.opendocument.load(str(path))
    parts: List[str] = []
    for paragraph in doc.getElementsByType(odf.text.P):
        parts.append(str(paragraph))
    return "\n".join(parts)


def _extract_doc_legacy(path: Path) -> str:
    try:
        import textract
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("textract is required for .doc/.ppt files") from exc
    data = textract.process(str(path))
    return data.decode("utf-8", errors="replace")


def _extract_image(path: Path) -> str:
    image = Image.open(path)
    return _ocr_image(image)


def _extract_eml(path: Path, temp_dir: Path) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    parts: List[str] = []
    subject = msg.get("subject", "")
    if subject:
        parts.append(f"Subject: {subject}")
    from_header = msg.get("from", "")
    if from_header:
        parts.append(f"From: {from_header}")
    to_header = msg.get("to", "")
    if to_header:
        parts.append(f"To: {to_header}")

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = part.get_content_disposition()
            if content_disposition == "attachment":
                filename = Path(part.get_filename() or "attachment").name
                payload = part.get_payload(decode=True) or b""
                attachment_path = temp_dir / filename
                attachment_path.write_bytes(payload)
                parts.append(_extract_file(attachment_path, temp_dir))
            elif part.get_content_type().startswith("text/"):
                text = part.get_content()
                if text:
                    parts.append(text)
            elif part.get_content_type().startswith("image/"):
                filename = Path(part.get_filename() or "inline-image").name
                payload = part.get_payload(decode=True) or b""
                attachment_path = temp_dir / filename
                attachment_path.write_bytes(payload)
                parts.append(_extract_file(attachment_path, temp_dir))
    else:
        text = msg.get_content()
        if text:
            parts.append(text)

    return "\n".join([p for p in parts if p])


def _extract_msg(path: Path, temp_dir: Path) -> str:
    import extract_msg

    msg = extract_msg.Message(str(path))
    msg.process()
    parts: List[str] = []
    if msg.subject:
        parts.append(f"Subject: {msg.subject}")
    if msg.sender:
        parts.append(f"From: {msg.sender}")
    if msg.to:
        parts.append(f"To: {msg.to}")
    if msg.body:
        parts.append(msg.body)

    for attachment in msg.attachments:
        filename = Path(attachment.longFilename or attachment.shortFilename or "attachment").name
        attachment_path = temp_dir / filename
        attachment.save(customPath=str(temp_dir))
        if attachment_path.exists():
            parts.append(_extract_file(attachment_path, temp_dir))

    return "\n".join([p for p in parts if p])


def _extract_archive(path: Path, temp_dir: Path) -> str:
    extract_dir = temp_dir / "archive"
    extract_dir.mkdir(parents=True, exist_ok=True)
    max_files = 2000
    max_bytes = 200 * 1024 * 1024

    if path.suffix.lower() == ".zip":
        _extract_zip(path, extract_dir, max_files, max_bytes)
    elif path.suffix.lower() == ".7z":
        _extract_7z(path, extract_dir, max_files, max_bytes)
    elif path.suffix.lower() == ".rar":
        _extract_rar(path, extract_dir, max_files, max_bytes)
    else:
        raise ValueError("Unsupported archive")

    parts: List[str] = []
    for file in extract_dir.rglob("*"):
        if file.is_file():
            parts.append(_extract_file(file, temp_dir))
    return "\n".join([p for p in parts if p])


def _extract_file(path: Path, temp_dir: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".doc":
        raise ValueError("Legacy .doc files are not supported in this build")
    if suffix == ".pptx":
        return _extract_pptx(path)
    if suffix == ".ppt":
        raise ValueError("Legacy .ppt files are not supported in this build")
    if suffix == ".xlsx":
        return _extract_xlsx(path)
    if suffix == ".xls":
        return _extract_xls(path)
    if suffix == ".rtf":
        return _extract_rtf(path)
    if suffix == ".odt":
        return _extract_odt(path)
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"}:
        return _extract_image(path)
    if suffix == ".eml":
        return _extract_eml(path, temp_dir)
    if suffix == ".msg":
        return _extract_msg(path, temp_dir)
    if suffix in {".zip", ".7z", ".rar"}:
        return _extract_archive(path, temp_dir)

    raise ValueError(f"Unsupported file type: {suffix}")


def extract_text_from_uploads(uploaded_files: Iterable) -> Tuple[str, List[ExtractionResult]]:
    combined_parts: List[str] = []
    results: List[ExtractionResult] = []

    for uploaded in uploaded_files:
        filename = Path(uploaded.name).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            results.append(ExtractionResult(filename=filename, ok=False, message="Unsupported file type"))
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            file_path = temp_dir / filename
            file_path.write_bytes(uploaded.getvalue())
            try:
                text = _extract_file(file_path, temp_dir)
                combined_parts.append(f"===== FILE: {filename} =====\n{text}\n")
                results.append(ExtractionResult(filename=filename, ok=True, message="Extracted"))
            except Exception as exc:
                results.append(ExtractionResult(filename=filename, ok=False, message=str(exc)))

    return "\n".join(combined_parts).strip(), results
