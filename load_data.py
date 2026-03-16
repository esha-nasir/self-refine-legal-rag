import os
import time
import base64
import requests
from io import BytesIO
from urllib.parse import urljoin

import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

from chunking import chunk_text

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# ------------------- ENV / CONFIG -------------------
PDF_FOLDER = os.getenv("PDF_FOLDER", "/path/to/pdfs")
CSV_PATH = os.getenv("CSV_PATH", "/path/to/judgments.csv")

# ✅ Prefix for links stored in CSV (temp_link)
BASE_URL = os.getenv("BASE_URL", "https://api.sci.gov.in/")

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en,fr,es,pt").split(",")
LOCAL_OCR_LANG = os.getenv("LOCAL_OCR_LANG", "eng")
LOCAL_TEXT_MIN_WORDS = int(os.getenv("LOCAL_TEXT_MIN_WORDS", "80"))
USE_LOCAL_OCR_FALLBACK = os.getenv("USE_LOCAL_OCR_FALLBACK", "1").strip() == "1"

MAX_DPI = int(os.getenv("MAX_DPI", "300"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(10000 * 10000)))
SLEEP_BETWEEN_REQUESTS = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.5"))
RETRIES = int(os.getenv("RETRIES", "3"))

if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV not found: {CSV_PATH}")

if not os.path.exists(PDF_FOLDER):
    raise RuntimeError(f"PDF folder not found: {PDF_FOLDER}")

# Old Yandex-only validation kept commented for reference.
# if not FOLDER_ID:
#     raise RuntimeError("Missing env var: YANDEX_FOLDER_ID")
# if not YANDEX_API_KEY and not YANDEX_IAM_TOKEN:
#     raise RuntimeError("Set one of: YANDEX_API_KEY or YANDEX_IAM_TOKEN")

# ------------------- LOAD CSV -------------------
csv_df = pd.read_csv(
    CSV_PATH,
    sep=";",
    engine="python",
    on_bad_lines="warn",
)


# ------------------- HELPERS -------------------
def clean_value(v):
    """
    Pinecone metadata cannot contain None / NaN.
    Convert None/NaN to "" (empty string).
    """
    if v is None or pd.isna(v):
        return ""
    if isinstance(v,str):
        return v.strip()
    return v


def build_full_url(temp_link_value) -> str:
    """
    CSV temp_link looks like: "supremecourt//19/19__Judgement_10-Apr-2019.pdf"
    We want: "https://api.sci.gov.in/supremecourt/19/19__Judgement_10-Apr-2019.pdf"
    """
    raw = clean_value(temp_link_value)
    if not raw:
        return ""

    raw = str(raw).strip()

    # If CSV already contains a full URL, keep it
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw

    # Normalize slashes a bit
    raw = raw.lstrip("/")          # remove leading /
    while "//" in raw:
        raw = raw.replace("//", "/")

    return urljoin(BASE_URL, raw)


# ------------------- OCR STRATEGIES -------------------
# Old Yandex OCR logic kept commented for reference.
# def yandex_ocr_image(img, languages=OCR_LANGUAGES) -> str:
#     try:
#         buffer = BytesIO()
#         img.save(buffer, format="JPEG")
#         encoded = base64.b64encode(buffer.getvalue()).decode()
#
#         url = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
#         auth_value = f"Bearer {YANDEX_IAM_TOKEN}" if YANDEX_IAM_TOKEN else f"Api-Key {YANDEX_API_KEY}"
#         headers = {
#             "Authorization": auth_value,
#             "Content-Type": "application/json",
#         }
#
#         data = {
#             "folderId": FOLDER_ID,
#             "analyze_specs": [
#                 {
#                     "content": encoded,
#                     "features": [
#                         {
#                             "type": "TEXT_DETECTION",
#                             "text_detection_config": {"language_codes": languages},
#                         }
#                     ],
#                 }
#             ],
#         }
#
#         response = requests.post(url, headers=headers, json=data, timeout=60)
#         response.raise_for_status()
#         result = response.json()
#
#         pages = (
#             result.get("results", [{}])[0]
#             .get("results", [{}])[0]
#             .get("textDetection", {})
#             .get("pages", [])
#         )
#
#         lines_out = []
#         for page in pages:
#             for block in page.get("blocks", []):
#                 for line in block.get("lines", []):
#                     words = [w.get("text", "") for w in line.get("words", [])]
#                     line_txt = " ".join([w for w in words if w])
#                     if line_txt.strip():
#                         lines_out.append(line_txt)
#
#         return "\n".join(lines_out).strip()
#
#     except Exception as e:
#         print(f"❌ Yandex OCR failed: {e}")
#         return ""


def local_ocr_image(img) -> str:
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=LOCAL_OCR_LANG).strip()
    except Exception as e:
        print(f"❌ Local OCR failed: {e}")
        return ""


def extract_text_native(pdf_path: str) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            text = (page.get_text("text") or "").strip()
            if text:
                texts.append(text)
        return "\n\n".join(texts).strip()
    except Exception as e:
        print(f"❌ Native PDF extraction failed for {pdf_path}: {e}")
        return ""


def _looks_like_good_text(text: str) -> bool:
    return len((text or "").split()) >= LOCAL_TEXT_MIN_WORDS


# ------------------- PDF TO TEXT -------------------
def extract_text_via_local_ocr(pdf_path: str) -> str:
    text = ""
    try:
        images = convert_from_path(pdf_path, dpi=MAX_DPI)

        for i, img in enumerate(images, 1):
            print(f"   📄 OCR page {i}: {os.path.basename(pdf_path)}")

            # downscale if huge
            if img.width * img.height > MAX_PIXELS:
                scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.LANCZOS,
                )

            for attempt in range(1, RETRIES + 1):
                page_text = local_ocr_image(img)

                if page_text.strip():
                    text += page_text + "\n"
                    print(f"      ✔ Extracted {len(page_text.split())} words")
                    break
                elif attempt < RETRIES:
                    time.sleep(2**attempt)
                else:
                    print(f"      ⚠ OCR failed after {RETRIES} attempts")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except Exception as e:
        print(f"❌ Failed to process {pdf_path}: {e}")

    return text.strip()


def extract_text_smart(pdf_path: str) -> str:
    native_text = extract_text_native(pdf_path)
    if _looks_like_good_text(native_text):
        print(f"   ✔ Native text extraction used: {os.path.basename(pdf_path)}")
        return native_text

    if USE_LOCAL_OCR_FALLBACK:
        print(f"   🔁 Falling back to local OCR: {os.path.basename(pdf_path)}")
        ocr_text = extract_text_via_local_ocr(pdf_path)
        if ocr_text.strip():
            return ocr_text

    return native_text.strip()


# ------------------- LOAD DOCUMENTS -------------------
def load_documents(pdf_folder: str = PDF_FOLDER):
    print("🔥 load_documents() started")
    docs = []

    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(root, file)
            print(f"➡ Processing PDF: {file}")

            text = extract_text_smart(pdf_path)
            if not text.strip():
                print(f"⚠ Skipping empty PDF: {file}")
                continue

            # Match CSV metadata
            if "temp_link" in csv_df.columns:
                row = csv_df[csv_df["temp_link"].astype(str).str.endswith(file)]
            else:
                row = pd.DataFrame()

            if row.empty:
                print(f"⚠ CSV metadata not found for {file}")
                metadata = {
                    "file_name": file,
                    "folder": os.path.relpath(root, pdf_folder),
                }
            else:
                r = row.iloc[0]
                metadata = {
                    "diary_no": clean_value(r.get("diary_no", "")),
                    "judgement_type": clean_value(r.get("Judgement_type", "")),
                    "case_no": clean_value(r.get("case_no", "")),
                    "pet": clean_value(r.get("pet", "")),
                    "res": clean_value(r.get("res", "")),
                    "pet_adv": clean_value(r.get("pet_adv", "")),
                    "res_adv": clean_value(r.get("res_adv", "")),
                    "bench": clean_value(r.get("bench", "")),
                    "judgement_by": clean_value(r.get("judgement_by", "")),
                    "judgment_dates": clean_value(r.get("judgment_dates", "")),
                    # ✅ Build full URL with https://api.sci.gov.in/ prefix
                    "url": build_full_url(r.get("temp_link", "")),
                    "file_name": file,
                    "folder": os.path.relpath(root, pdf_folder),
                }

            chunks = chunk_text(text, chunk_size=450, overlap=80)
            print(f"🧩 {file}: {len(chunks)} chunks")

            docs.append({"text": text, "metadata": metadata, "chunks": chunks})

    print(f"✅ Loaded {len(docs)} PDFs")
    return docs
