# Historical Document Data Extraction Benchmark

A comprehensive benchmarking pipeline for comparing different approaches to extracting structured data from historical documents. This repository provides tools to evaluate various OCR and Large Language Model (LLM) based approaches for converting historical documents into OCR-generated text files and also into structured data formats.

## 📚 Overview

This pipeline is specifically designed for economic historians working with historical documents. It compares different methods of extracting structured data (in CSV format) from historical documents, including:

1. Traditional OCR approaches to generate OCR-generated text files:
   - Tesseract OCR (open-source)
   - Transkribus (commercial)

2. Modern LLM-based approaches to generate OCR-generated text files and also structured CSV data formats:
   - GPT-4o (OpenAI)
   - o1 (OpenAI; first reasoning model with image input)
   - Gemini-2.0-flash (Google)

The pipeline has different modes that are all benchmarked against the same ground truth TXT and CSV files:
- PDF with multiple pages → TXT/CSV
- A single PNG → TXT/CSV
- TXT → CSV

## 🎯 Key Features

- Multiple OCR and LLM-based text extraction methods
- Structured data extraction to CSV format
- Comprehensive benchmarking against TXT and CSV ground truth created and refined by the authors
- Scalable architecture for adding new models

## 📋 Requirements

### System Requirements
- Conda package manager
- Sufficient disk space for document processing, especially for long documents

### External Dependencies
- Tesseract OCR engine
- API keys for:
  - OpenAI (GPT-4o)
  - OpenAI (o1)
  - Google (Gemini-2.0-flash)
  - Transkribus

## 🚀 Installation

1. Clone the repository by running the following command in your terminal:
```bash
git clone https://github.com/yourusername/llm_historical_dataset_benchmarking.git
cd llm_historical_dataset_benchmarking
```

2. Create and activate the Conda environment. You have to install conda first:
```bash
conda env create -f config/environment.yml
conda activate llm_historical_dataset_benchmarking_env
```

3. Set up your API keys in `config/.env`:
```bash
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
TRANSKRIBUS_USERNAME=your_username
TRANSKRIBUS_PASSWORD=your_password
TRANSKRIBUS_LINE_DETECTION_ID=your_id
TRANSKRIBUS_HTR_ID=your_id
```

## 📁 Project Structure

```
.
├── config/               # Configuration files
│   ├── environment.yml   # Conda environment specification
│   └── .env              # API keys and credentials
├── data/
│   ├── pdfs/             # Input PDFs (type-1.pdf to type-10.pdf)
│   ├── ground_truth/     # Ground truth CSV files
│   └── page_by_page/     # Intermediate image files
├── results/              # Output directory for all models
├── src/                  # Source code
│   ├── benchmarking/     # Benchmarking tools
│   ├── llm_img2csv/      # Image to CSV converters
│   ├── llm_pdf2csv/      # PDF to CSV converters
│   ├── llm_txt2csv/      # Text to CSV converters
│   ├── ocr_img2txt/      # OCR processors
│   └── scripts/          # Utility scripts
└── logs/                 # Log files
```

## 🔧 Usage

### Basic Usage

```bash
# For Gemini 2.0
python src/llm_img2csv/gemini-2.0.py --pdf type-1.pdf

# For GPT-4o
python src/llm_img2csv/gpt-4o.py --pdf type-1.pdf

# Run all pdfs for all models
python src/scripts/all_llm_img2csv.py --pdfs type-1.pdf type-2.pdf type-3.pdf ... --models gemini-2.0 gpt-4o

# For Tesseract OCR
python src/ocr_img2txt/pytesseractOCR.py --pdf type-1.pdf

# For Transkribus OCR
python src/ocr_img2txt/transkribusOCR.py --pdf type-1.pdf

# Generate benchmarking results after running all models:
python src/benchmarking/csv_accuracy.py

# Run --help for more information:
python src/llm_img2csv/gemini-2.0.py --help
```

### Input Data Format

The pipeline expects:
- PDFs in `data/pdfs/` For the benchmarking, they are named as `type-N.pdf` (where N is 1-10). They can have other names.
- Ground truth CSV files in `data/ground_truth/csv/`
- Each CSV contains 4 columns that were extracted from the PDFs:
  - "first and middle names"
  - "surname"
  - "occupation"
  - "address"
- Prompts in `src/prompts/` for each LLM-based approach

### Output Format

Results are organized by:
1. Model type and function (llm_img2csv, llm_pdf2csv, llm_txt2csv, ocr_img2txt)
2. Temperature setting (for LLMs, benchmarking default is 0.0)
3. Run number (01, 02, 03, benchmarking default is run_01)

Example path:
```
results/llm_img2csv/<model_name>/<pdf_name_stem>/temperature_0.0/run_01/<pdf_name_stem>.csv
```

So for type-1.pdf and gpt-4o, the path is:
```
results/llm_img2csv/gpt-4o/type-1/temperature_0.0/run_01/type-1.csv
```

## 📊 Benchmarking

The benchmarking system evaluates:
1. **Exact Matching**: Character-for-character accuracy
2. **Normalized Matching**: Case-insensitive, punctuation-normalized comparison
3. **Fuzzy Matching**: Using Jaro-Winkler similarity (threshold: 0.90)

Results are generated as:
- CSV files with detailed metrics
- HTML reports with visual comparisons
- Aggregated statistics by document type and field

## 🙏 Acknowledgments

- OpenAI
- Google
- Transkribus team
- Tesseract OCR community

## 👥 Contributing

Contributions are very welcome! Feel free to:
- Fork the repository
- Create a feature branch
- Submit pull requests
- Open issues for bugs or feature requests
- Suggest improvements to documentation

Please ensure your contributions align with the project's coding standards and include appropriate tests where applicable.

## 📝 How to Cite

If you use this pipeline in your research, please cite:

```bibtex
@article{gg2025historicalbenchmarking,
    title={Benchmarking Large Language Models and OCR Systems for Historical Document Data Extraction: A Comprehensive Pipeline},
    author={Griesshaber, Niclas and Greif, Gavin},
    year={2025},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/forthcoming}
}
```

## 📧 Contact

Niclas Griesshaber, niclas.griesshaber@history.ox.ac.uk

Gavin Greif, gavin.greif@history.ox.ac.uk