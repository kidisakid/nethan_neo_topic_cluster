"""
Topic Clustering Streamlit Application

A web-based interface for clustering text data into topics using cosine similarity
and K-Means clustering. Upload a CSV file, configure clustering parameters, and
analyze topic clusters.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Add clustering module to path
try:
    from clustering.topic_clustering import TopicClusterer, perform_topic_clustering, clean_text
except ImportError:
    st.error("Cannot import clustering module. Make sure the 'clustering' folder is in the same directory.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Topic Clustering",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
    }
    .section-header {
        color: #2ca02c;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_clustered' not in st.session_state:
        st.session_state.df_clustered = None
    if 'clusterer' not in st.session_state:
        st.session_state.clusterer = None


def load_data(uploaded_file):
    """Load CSV data from uploaded file with multiple encoding support."""
    # List of encodings to try, ordered by likelihood
    csv_encodings = [
        'utf-8',
        'iso-8859-1',
        'cp1252',
        'utf-16',
        'latin-1',
        'ascii',
        'cp1251',
        'gb2312',
        'gbk',
    ]
    
    csv_delimiters = [',', ';', '\t', '|', ' ']
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = None
            last_error = None
            
            # Try different encoding and delimiter combinations
            for encoding in csv_encodings:
                for delimiter in csv_delimiters:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(
                            uploaded_file,
                            encoding=encoding,
                            sep=delimiter,
                            on_bad_lines='skip',
                            engine='python'
                        )
                        # Check if we got meaningful columns
                        if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                            return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        last_error = f"encoding={encoding}, delimiter={repr(delimiter)}"
                        continue
            
            # If we get here, try Excel format as fallback
            if uploaded_file.name.endswith(('.xlsx', '.xls', '.csv')):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    pass
            
            if df is None:
                st.error(
                    f"Unable to read file with tried encodings: {', '.join(csv_encodings[:3])}... "
                    f"and delimiters: {', '.join(repr(d) for d in csv_delimiters[:3])}...\n\n"
                    f"Last attempt: {last_error}"
                )
            return df
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return df
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}\n\nPlease ensure your file is a valid CSV or Excel file.")
        return None


def display_header():
    """Display application header."""
    st.markdown('<div class="main-header">🔍 Topic Clustering Application</div>', unsafe_allow_html=True)


def display_file_upload_section():
    """Display file upload section."""
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    # Add card CSS
    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 2px 8px rgba(31, 119, 180, 0.08);
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card-label {
            font-size: 0.72em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .metric-card-value {
            font-size: 2em;
            font-weight: 700;
            color: #1f77b4;
            line-height: 1.1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum size: 200MB"
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df_original = df
            st.markdown('<div class="success-box">✅ File loaded successfully!</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Rows</div>
                    <div class="metric-card-value">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Columns</div>
                    <div class="metric-card-value">{len(df.columns):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
    
    return st.session_state.df_original


def display_column_selection(df):
    """Display column selection section."""
    if df is None:
        return None
    
    st.markdown('<div class="section-header">🎯 Column Selection</div>', unsafe_allow_html=True)
    
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.warning("No text columns found in the dataset.")
        return None
    
    selected_column = st.selectbox(
        "Select the column to cluster:",
        text_columns,
        help="Choose the column containing text data"
    )
    
    # Display column statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Non-null values", df[selected_column].notna().sum())
    with col2:
        st.metric("Null values", df[selected_column].isna().sum())
    with col3:
        avg_length = df[selected_column].astype(str).str.len().mean()
        st.metric("Avg text length", f"{int(avg_length)} chars")
    
    return selected_column


def display_clustering_parameters():
    """Display clustering parameter configuration section."""
    st.markdown('<div class="section-header">⚙️ Clustering Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="The number of topic clusters to create"
        )
    
    with col2:
        max_df = st.slider(
            "Max document frequency",
            min_value=0.1,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Ignore terms that appear in more than this fraction of documents"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_df = st.slider(
            "Min document frequency",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Ignore terms appearing in fewer documents than this"
        )
    
    with col4:
        n_init = st.slider(
            "K-Means initializations",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Number of times K-Means will run with different centroid seeds"
        )
    
    clean_text_option = st.checkbox(
        "Clean text before clustering",
        value=True,
        help="Remove URLs, punctuation, and extra whitespace"
    )
    
    return {
        'n_clusters': n_clusters,
        'max_df': max_df,
        'min_df': min_df,
        'n_init': n_init,
        'clean_text': clean_text_option,
    }


def perform_clustering(df, text_column, params):
    """Perform clustering with progress tracking."""
    try:
        texts = df[text_column].tolist()
        
        # Clean texts if requested
        if params['clean_text']:
            with st.spinner("Cleaning text..."):
                texts = [clean_text(str(t)) for t in texts]
        
        # Create clusterer
        clusterer = TopicClusterer(
            n_clusters=params['n_clusters'],
            max_df=params['max_df'],
            min_df=params['min_df'],
            n_init=params['n_init'],
        )
        
        # Perform clustering
        with st.spinner("Vectorizing text and clustering..."):
            clusters = clusterer.fit_predict(texts, show_progress=False)
        
        # Get additional information
        top_terms = clusterer.get_top_terms(n_terms=10)
        distances = clusterer.get_cluster_distances(texts)
        
        # Create result dataframe
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        st.session_state.df_clustered = result_df
        st.session_state.clusterer = clusterer
        
        return result_df, top_terms, distances
    
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        return None, None, None


def display_clustering_results(result_df, top_terms, distances, n_clusters):
    """Display clustering results."""
    st.divider()
    st.markdown('<div class="section-header">📊 Clustering Results</div>', unsafe_allow_html=True)
    
    # Cluster distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Cluster Distribution:**")
        cluster_counts = result_df['Cluster'].value_counts().sort_index()
        st.dataframe(
            cluster_counts.rename('Count'),
            use_container_width=True
        )
    
    with col2:
        if cluster_counts is not None:
            st.bar_chart(cluster_counts)
    
    st.divider()
    
    # Top terms per cluster
    st.markdown("### Top Terms per Cluster:")
    
    cols = st.columns(min(3, n_clusters))
    for cluster_id in range(n_clusters):
        with cols[cluster_id % len(cols)]:
            terms = top_terms.get(cluster_id, [])
            st.markdown(f"**Cluster {cluster_id}**")
            for i, term in enumerate(terms[:5], 1):
                st.write(f"{i}. {term}")
    
    st.divider()
    
    # Display clustered data
    st.markdown("### Clustered Data Preview:")
    
    cluster_filter = st.multiselect(
        "Filter by cluster(s):",
        sorted(result_df['Cluster'].unique()),
        default=sorted(result_df['Cluster'].unique())
    )
    
    filtered_df = result_df[result_df['Cluster'].isin(cluster_filter)]
    st.dataframe(filtered_df, use_container_width=True)


def display_detailed_analysis(clusterer, result_df):
    """Display detailed cluster analysis."""
    st.markdown('<div class="section-header">🔬 Detailed Cluster Analysis</div>', unsafe_allow_html=True)
    
    if clusterer is None:
        st.warning("No clustering results available.")
        return
    
    # Cluster statistics
    st.markdown("**Cluster Statistics:**")
    
    cluster_stats = []
    for cluster_id in range(clusterer.n_clusters):
        cluster_data = result_df[result_df['Cluster'] == cluster_id]
        cluster_stats.append({
            'Cluster': cluster_id,
            'Documents': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(result_df) * 100):.1f}%"
        })
    
    stats_df = pd.DataFrame(cluster_stats)
    st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    # Cluster visualization
    st.markdown("**Cluster Quality Metrics:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inertia = clusterer.kmeans.inertia_
        st.metric("Inertia (Sum of Squared Distances)", f"{inertia:.2f}")
    
    with col2:
        silhouette_samples = np.sqrt(inertia / len(result_df))
        st.metric("Average Distance to Centroid", f"{silhouette_samples:.2f}")


def display_download_section(df):
    """Display download section."""
    if df is None or df.empty:
        return
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv",
        help="Download the clustered data as CSV",
        use_container_width=True,
        type="primary"
    )

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    display_header()
    
    # File upload
    df = display_file_upload_section()
    
    if df is not None:
        # Column selection
        text_column = display_column_selection(df)
        
        if text_column is not None:
            # Clustering parameters
            params = display_clustering_parameters()
            
            # Clustering button
            if st.button("🚀 Start Clustering", use_container_width=True, type="primary"):
                result_df, top_terms, distances = perform_clustering(df, text_column, params)
                
                if result_df is not None:
                    st.markdown('<div class="success-box">✅ Clustering completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Display results
                    display_clustering_results(result_df, top_terms, distances, params['n_clusters'])
                    
                    # Detailed analysis
                    display_detailed_analysis(st.session_state.clusterer, result_df)
                    
                    # Download section
                    display_download_section(result_df)
            
            # Display previous results if available
            elif st.session_state.df_clustered is not None:
                st.markdown('<div class="info-box">ℹ️ Showing previous clustering results. Click "Start Clustering" to update.</div>', unsafe_allow_html=True)
                
                result_df = st.session_state.df_clustered
                clusterer = st.session_state.clusterer
                top_terms = clusterer.get_top_terms(n_terms=10)
                distances = clusterer.get_cluster_distances(result_df[text_column].tolist())
                
                display_clustering_results(result_df, top_terms, distances, params['n_clusters'])
                display_detailed_analysis(clusterer, result_df)
                display_download_section(result_df)

if __name__ == "__main__":
    main()
