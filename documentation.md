# Running the Streamlit Application

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The app will open in your default browser at `http://localhost:8501`

## Application Features

### 📁 Data Upload
- Upload CSV or Excel files
- Automatic file validation
- Display file statistics (rows, columns, size)
- Preview data in the application

### 🎯 Column Selection
- Select which text column to cluster
- View column statistics:
  - Number of non-null values
  - Number of null values
  - Average text length

### ⚙️ Clustering Configuration
- **Number of clusters** (2-20): Controls how many topics to create
- **Max document frequency** (0.1-1.0): Filter out very common terms
- **Min document frequency** (1-10): Filter out rare terms
- **K-Means initializations** (1-20): More initializations = more reliable but slower
- **Clean text** option: Optionally clean URLs, punctuation, and whitespace

### 📊 Clustering Results
- **Cluster Distribution**: See how many documents are in each cluster
- **Top Terms**: View the most representative words for each cluster
- **Cluster Filter**: Filter and view documents by specific clusters
- **Full Data Preview**: Interactive table showing all data with cluster assignments

### 🔬 Detailed Analysis
- **Cluster Statistics**: Document counts and percentages per cluster
- **Cluster Quality Metrics**: Inertia and average distances

### 💾 Download Results
- **Download as CSV**: Save results as a CSV file
- **Download as Excel**: Save results as an Excel file (.xlsx)

## Parameter Guide

### Number of Clusters
- **Low values (2-3)**: Fewer, broader topics
- **Medium values (5-10)**: Balanced topic granularity (recommended)
- **High values (15+)**: Many specific topics, but less statistical power

### Document Frequency Settings
- **max_df = 0.95**: Allows very common terms (less filtering)
- **max_df = 0.5**: More aggressive filtering of common terms
- **min_df = 1**: Keep all terms that appear at least once
- **min_df = 5**: Remove very rare terms

**Recommendations:**
- For small datasets (< 1000 documents): Use higher min_df
- For large datasets: Default values work well

### K-Means Initializations
- **Lower values (1-5)**: Faster but potentially less stable
- **Default (10)**: Good balance between speed and stability
- **Higher values (15+)**: Most stable but slower

## File Format Requirements

### CSV Files
- Standard CSV with comma, semicolon, or pipe delimiters
- UTF-8 encoding recommended
- Maximum 200MB

### Excel Files
- .xlsx or .xls format
- Single sheet or first sheet will be used

## Example Workflow

1. **Prepare Data**: Have a CSV or Excel file with a text column
2. **Open App**: Run `python -m streamlit run app.py` or use run scripts
3. **Upload File**: Click to upload your file
4. **Select Column**: Choose the column containing text
5. **Configure Parameters**: Adjust clustering settings (or use defaults)
6. **Start Clustering**: Click the "Start Clustering" button
7. **Review Results**: Explore clusters, top terms, and statistics
8. **Download**: Save results as CSV or Excel

## Troubleshooting

### Port Already in Use
If port 8501 is already used, run:
```bash
streamlit run app.py --server.port 8502
```

### File Upload Error
- Check file format (CSV or Excel only)
- Verify file is not corrupted
- Try a smaller sample of your data

### Memory Issues
- Use min_df parameter to remove rare terms
- Reduce max_features in clustering/topic_clustering.py
- Work with a sample of your data first

### Slow Performance
- Reduce number of clusters
- Increase min_df to filter rare terms
- Reduce K-Means initializations
- Use a smaller dataset

## Keyboard Shortcuts

- **Ctrl+C**: Stop the Streamlit server
- **R**: Rerun the app (in browser)

## Tips for Best Results

1. **Clean your data first**: Though the app has cleaning options, pre-cleaned data produces better results
2. **Choose the right number of clusters**: Start with 5 and adjust based on results
3. **Review top terms**: Check if clusters make semantic sense
4. **Adjust document frequency**: Filter noise while keeping meaningful terms
5. **Experiment**: Try different parameter combinations to find what works best

## Screen Layout

The app is organized into logical sections:

```
Header and Description
├── Data Upload Section
│   ├── File uploader
│   └── File preview
├── Column Selection Section
│   ├── Column dropdown
│   └── Column statistics
├── Clustering Parameters Section
│   ├── Parameter sliders
│   └── Text cleaning option
├── Results Section (after clustering)
│   ├── Cluster distribution
│   ├── Top terms
│   └── Filtered data view
├── Detailed Analysis Section
│   ├── Cluster statistics
│   └── Quality metrics
└── Download Section
    ├── CSV download
    └── Excel download
```

## Performance Notes

- **Small datasets** (< 1000 docs): < 1 second
- **Medium datasets** (1K-10K docs): 1-10 seconds
- **Large datasets** (10K+ docs): 10-60+ seconds

Processing time depends on:
- Number of documents
- Average document length
- Number of clusters
- Number of K-Means initializations

## Contact & Support

For issues or improvements to the clustering module, refer to the main README.md

Enjoy clustering! 🚀
