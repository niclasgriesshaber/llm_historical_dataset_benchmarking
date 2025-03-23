# Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents

This repository provides comprehensive benchmarks to evaluate conventional OCR pipelines and multimodal Large Language Model (mLLM) for converting historical documents into OCR-generated text files and also into CSV datasets.

## 📚 Overview

This pipeline is specifically designed for economic historians working with historical documents. It compares different methods of extracting structured data (in CSV format) from historical documents, including:

1. Traditional OCR approaches to generate OCR-generated text files:
   - Tesseract OCR (open-source; "deu", "deu_frak")
   - Transkribus (commercial; "Text Titan I", "Print M1")

2. Multimodal LLMs for OCR, OCR post-correction, and structured data extraction:
   - GPT-4o (OpenAI)
   - Gemini-2.0-flash (Google)

The pipeline has different modes that are all benchmarked against the same ground truth TXT and CSV files:
- A single PNGs → TXT/CSV
- A single PNG + Transkribus OCR → TXT
- A PDF with 3 pages → TXT/CSV
- TXT → CSV

## 🎯 Key Features

- Multiple OCR systems for text extraction
- OCR post-correction using multimodal LLMs
- Named entity recognition to construct CSV datasets using multimodal LLMs
- Comprehensive benchmarking against TXT and CSV ground truth created and refined by the authors
- Scalable architecture for adding new mLLMs

## 📋 Requirements

### API Keys or Credentials for:
- OpenAI (GPT-4o)
- Google (Gemini-2.0-flash)
- Transkribus

## 🚀 Installation

1. Clone the repository by running the following command in your terminal:
```bash
git clone https://github.com/niclasgriesshaber/llm_historical_dataset_pipeline_benchmarking.git
cd llm_historical_dataset_pipeline_benchmarking
```

2. Install the Conda package manager (for historians new to programming):
   
   a. Download Miniconda (a minimal version of Conda):
      - For macOS: Visit https://docs.conda.io/en/latest/miniconda.html and download the macOS installer
      - For Windows: Visit https://docs.conda.io/en/latest/miniconda.html and download the Windows installer
      - For Linux: Visit https://docs.conda.io/en/latest/miniconda.html and download the Linux installer
   
   b. Install Miniconda:
      - macOS/Linux: Open Terminal, navigate to the download location, and run `bash Miniconda3-latest-*.sh`
      - Windows: Double-click the downloaded installer and follow the prompts
   
   c. Verify installation by opening a new terminal window and typing `conda --version`

3. Create and activate the Conda environment:
```bash
conda env create -f config/environment.yml
conda activate llm_historical_dataset_benchmarking_env
```

4. Set up your API keys in `config/.env`:
```
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
TRANSKRIBUS_USERNAME=your_username
TRANSKRIBUS_PASSWORD=your_password
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
│   └── page_by_page/     # Intermediate image files as single PNGs
├── results/              # Output directory for all models
├── src/                  # Source code
│   ├── benchmarking/     # Benchmarking tools
│   ├── llm_img2csv/      # Image to CSV converters using multimodal LLMs
│   ├── llm_pdf2csv/      # PDF to CSV converters using multimodal LLMs
│   ├── llm_txt2csv/      # Text to CSV converters
│   ├── ocr_img2txt/      # OCR processors
│   └── scripts/          # Utility scripts
└── logs/                 # Log files
```

## 🔧 Usage

### Basic Usage

```bash
# For multimodal LLM Gemini 2.0
python src/llm_img2csv/gemini-2.0.py --pdf type-1.pdf

# For multimodal LLM GPT-4o
python src/llm_img2csv/gpt-4o.py --pdf type-1.pdf

# Run all pdfs for all multimodal LLM models
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
- Prompts in `src/prompts/` for each multimodal LLM

### Output Format

Results are organized by:
1. Model type and function (llm_img2csv, llm_pdf2csv, llm_txt2csv, ocr_img2txt)
2. Temperature setting (for multimodal LLMs, benchmarking default is 0.0)
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
1. **Exact Matching**: Lower-cased, character-by-character accuracy
2. **Fuzzy Matching**: Using Jaro-Winkler similarity (threshold: 0.90), also lower-cased

Results are generated as:
- CSV files with detailed metrics
- HTML reports with visual comparisons
- Aggregated statistics by document type and field

## 🙏 Acknowledgments

- OpenAI Team
- Google Gemini Team
- Transkribus Team
- Tesseract OCR Community

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
@article{gg2025,
    title={Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents},
    author={Greif, Gavin and Griesshaber, Niclas},
    year={2025},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/forthcoming}
}
```

## 📧 Contact

Niclas Griesshaber, niclasgriesshaber@outlook.com
Gavin Greif, gavin.greif@history.ox.ac.uk