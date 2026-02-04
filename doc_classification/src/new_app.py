import streamlit as st
import re
import cv2
import pytesseract
import numpy as np
import pdfplumber
import mysql.connector
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# TESSERACT PATH
# =========================
pytesseract.pytesseract.tesseract_cmd = r"D:\users\tesseract-ocr\tesseract.exe"

# =========================
# DATABASE CONNECTION
# =========================
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="Rik2026@!?mysql",
            database="doc_ai",
            connection_timeout=5
        )
    except mysql.connector.Error as e:
        st.warning(f"Database not reachable: {e}")
        return None


def save_to_db(filename, text, prediction):
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (filename, document_text, predicted_type)
            VALUES (%s, %s, %s)
            """,
            (filename, text, prediction)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        st.warning(f"Failed to save to DB: {e}")

# =========================
# TRAINING DATA (FIXED) #training
# =========================

documents = [
    # Invoice
    ("invoice number invoice date total amount gst tax payable", "Invoice"),
    ("tax invoice bill amount due net total", "Invoice"),
    ("company name invoice subtotal cgst sgst grand total", "Invoice"),
    ("amount payable invoice thank you for your business", "Invoice"),

    # Resume
    ("education skills experience projects certifications", "Resume"),
    ("bachelor of engineering work experience objective", "Resume"),

    # Bank Statement
    ("account number bank statement debit credit balance", "Bank Statement"),
    ("opening balance closing balance transaction history", "Bank Statement"),

    # ID Proof
    ("government of india aadhaar dob gender", "ID Proof"),
    ("passport nationality date of birth", "ID Proof"),

    # Contract
    ("this agreement is made between the parties", "Contract"),
    ("terms and conditions legally binding agreement", "Contract"),
]

# =========================
# RULE-BASED KEYWORDS #classification
# =========================
DOC_KEYWORDS = {
    "Invoice": [
        "invoice", "INVOICE ", "invoice no", "gst", "cgst", "sgst", "basic invoice",
        "subtotal", "total due", "tax payable", "due date", "issued on"
    ],
    "Bank Statement": [
        "bank statement", "opening balance", "closing balance", "beginning balance", "withdrawals",
        "withdrawal", "deposit", "deposits", "statement period", "ending balance", "bank"
    ],
    "Resume": [
        "education", "skills", "experience", "projects","resume"
    ],
    "ID Proof": [
        "aadhaar", "passport", "date of birth", "gender", "proof",
        "government of"
    ],
    "Contract": [
        "contract", "CONTRACT","agreement", "terms and conditions",
        "legally binding", "hereby agree", "parties", "between", "signed by"
    ]
}



# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

texts = [clean_text(doc[0]) for doc in documents]
labels = [doc[1] for doc in documents]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# =========================
# SMART CLASSIFICATION
# =========================
INVOICE_KEYWORDS = [
    "invoice", "total", "amount", "gst", "tax", "bill", "payable"
]

def classify_document(text):
    cleaned = clean_text(text)

    if len(cleaned.split()) < 20:
        return "Unknown"

    keyword_scores = {}

    for doc_type, keywords in DOC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in cleaned)
        keyword_scores[doc_type] = hits

    best_doc = max(keyword_scores, key=keyword_scores.get)
    best_score = keyword_scores[best_doc]

    if best_score >= 2:
        return best_doc

    # ML fallback with confidence
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]
    max_prob = max(probs)

    if max_prob < 0.6:
        return "Unknown"

    return model.classes_[probs.argmax()]


# =========================
# OCR FUNCTIONS (CLEAN)
# =========================
def extract_text_from_image(uploaded_file):
    print("Processing file:", uploaded_file.name)

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Upscale
    gray = cv2.resize(
        gray, None,
        fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Noise removal
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=config)

    return text.strip()


def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Document Classification", layout="centered")

st.title("Document Classification System")
st.write("Upload Invoice, Resume, Bank Statement, ID Proof, or Contract")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    if uploaded_file.type.startswith("image"):
        text = extract_text_from_image(uploaded_file)

    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)

    else:
        st.error("Unsupported file type")
        st.stop()

    st.subheader("Extracted Text")
    st.text_area(
        label="Extracted Text",
        value=text,
        height=250
    )

    prediction = classify_document(text)

    st.subheader("Predicted Document Type")

    if prediction == "Unknown":
        st.warning("Document type could not be determined confidently")
    else:
        st.success(prediction)

    save_to_db(uploaded_file.name, text, prediction)

    st.info("Processing complete")
