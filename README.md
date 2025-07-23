# YouTube Viewership Analysis

Analyze YouTube viewership factors using **PCA**, **SVM**.

## 📂 Structure

```plaintext
metadata/              # Raw CSVs per channel (10 files)
output/                # Generated results and plots
.gitignore             # VCS ignore file
README.md              # Project overview
data_correction.py     # Fetches video metadata via YouTube API
data_analyze.py        # Preprocesses data, runs PCA, SVM analysis
requirements.txt       # Python dependencies
```

## ⚙️ Scripts

* **data\_correction.py**: Fetches video metadata (snippet, statistics, content details) via the YouTube Data API. Saves per-channel CSVs in the `metadata/` folder.
* **data\_analyze.py**: Loads all CSVs from `metadata/`, merges and preprocesses (parses durations, timestamps, counts tags, computes engagement rates), applies PCA, trains and evaluates SVM (classification/regression), then saves reports and plots in `output/`.

## 🚀 Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
2. **Fetch metadata**:

   ```bash
   python data_correction.py
   ```
3. **Run analysis**:

   ```bash
   python data_analyze.py
   ```

All intermediate results and final outputs (metrics and plots) will be available in the `output/` folder.

## 👥 Team Roles


* **Data Collection:** Kojima Yutaka, Yamasaki Aoto
* **Data Analysis:** Kojima Yutaka, Yamasaki Aoto
* **Report writer:** Nalishuwa Chama Joshua
* **Lead & Presenter:** Nigar Alizada
