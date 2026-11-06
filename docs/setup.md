# MIDAS Setup & Installation Guide

This guide details environment setup, configuration, and execution of the MIDAS pipeline.

---

## 1. System Requirements

- Python 3.11 or 3.12 (tested in CI)
- Windows, macOS, or Linux
- Optional: Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

---

## 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aT-404/MIDAS.git
cd MIDAS
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your Gemini API Key and settings:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
EXTRACTION_MODEL=gemini-2.0-flash
OCR_LANG=en
CONFIDENCE_THRESHOLD=0.75
DB_PATH=outputs/midas.db
```

> [!NOTE]
> If `GEMINI_API_KEY` is not provided, MIDAS automatically activates its high-precision deterministic heuristic extractor, ensuring 100% functionality offline.

---

## 4. Generate Synthetic Document Corpus

To generate the benchmark suite of 65 documents (30 invoices, 20 specs, 15 inspection reports) and degraded samples:

```bash
python data/scripts/generate_docs.py
python data/scripts/degrade.py
```

---

## 5. Running the Streamlit Dashboard

Launch the interactive UI:

```bash
streamlit run dashboard/app.py
```

Access the dashboard at `http://localhost:8501`.

---

## 6. Running via CLI

Process a single document:

```bash
python src/main.py --input data/samples/invoices/INV-0047.jpg --doc-type invoice
```

Search extracted records via SQLite FTS5:

```bash
python src/main.py --search "TATA METALIKS"
```

View system performance statistics:

```bash
python src/main.py --stats
```

---

## 7. Running the Test Suite

Execute the automated test suite with pytest:

```bash
pytest
```
