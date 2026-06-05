# Datasets for LLM Position Bias Analysis Framework

This directory contains the datasets used for analyzing position bias in LLM-based recommender systems. **Note: The actual data files are not included in this repository due to their large size (24+ GB total).**

## 📊 Supported Datasets

### 1. **Beauty Products Dataset**
- **Source**: Amazon Product Data (All_Beauty category)
- **Size**: ~759MB
- **Files**:
  - `All_Beauty.jsonl` - Product interaction data
  - `meta_All_Beauty.jsonl` - Product metadata
  - `user_title_timestamp.csv` - Processed user-item interactions
- **Description**: Beauty and personal care product reviews and ratings
- **Use Case**: Consumer product recommendation bias analysis

### 2. **Books Dataset** 
- **Source**: Amazon Product Data (Books category)
- **Size**: ~16GB
- **Files**:
  - `meta_Books.jsonl` - Book metadata (titles, authors, genres)
  - `ratings_Books.csv` - Book ratings and reviews
  - `user_title_timestamp.csv` - Processed user-book interactions
- **Description**: Book reviews, ratings, and metadata
- **Use Case**: Literature recommendation and reading preference analysis

### 3. **MovieLens-1M Dataset**
- **Source**: GroupLens Research (MovieLens 1M)
- **Size**: ~24MB
- **Files**:
  - `movies.dat` - Movie information (titles, genres)
  - `ratings.dat` - User ratings for movies (1-5 scale)
  - `users.dat` - User demographics (age, gender, occupation)
  - `README` - Original dataset documentation
- **Description**: Movie ratings with user demographics
- **Use Case**: Movie recommendation with demographic analysis

### 4. **Music Dataset (CDs & Vinyl)**
- **Source**: Amazon Product Data (CDs and Vinyl category)
- **Size**: ~2.8GB
- **Files**:
  - `meta_CDs_and_Vinyl.jsonl` - Music metadata
  - `ratings_CDs_and_Vinyl.csv` - Music ratings
  - `user_title_timestamp.csv` - Basic user-music interactions
  - `user_title_timestamp_metadata_*.csv` - Enhanced interaction data (versions 2-4)
- **Description**: Music album reviews and ratings
- **Use Case**: Music recommendation and preference analysis

### 5. **Steam Games Dataset**
- **Source**: Steam platform data
- **Size**: ~4.8GB
- **Files**:
  - `steam_games.json` - Game information and metadata
  - `steam_reviews.json` - User reviews and ratings
  - `user_title_timestamp.csv` - Basic user-game interactions
  - `user_title_timestamp_with_names.csv` - Enhanced with game names
  - `user_name_timestamp.csv` - User activity timestamps
- **Description**: Video game reviews and player behavior
- **Use Case**: Game recommendation and gaming preference analysis

## 🔄 Data Processing Pipeline

### Standard Format
All datasets are processed into a common format with columns:
- **UserID**: Unique user identifier
- **Title**: Item title/name
- **Rating**: User rating (when available)
- **Timestamp**: Interaction timestamp

### Processing Steps
1. **Raw Data Ingestion**: Load original dataset files
2. **Data Cleaning**: Remove duplicates, handle missing values
3. **Format Standardization**: Convert to common schema
4. **User Filtering**: Select users with sufficient interaction history (≥6 items)
5. **Quality Validation**: Ensure data integrity and consistency

## 📁 Directory Structure
```
data/
├── README.md                 # This file
├── beauty/                   # Beauty products (759MB)
├── books/                    # Books (16GB)
├── ml-1m/                    # MovieLens (24MB)
├── music/                    # Music/CDs (2.8GB)
├── news/                     # News articles (609MB)
└── steam/                    # Video games (4.8GB)
```

## 🚀 Getting the Data

### Download Instructions

#### 1. MovieLens-1M
```bash
# Download from GroupLens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
mv ml-1m/* data/ml-1m/
```

#### 2. Amazon Product Data
```bash
# Beauty Products 
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz

# Books
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz

# Music (CDs & Vinyl)
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/CDs_and_Vinyl.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_CDs_and_Vinyl.json.gz
```

#### 3. Steam Dataset
```bash
# Steam Games Metadata
wget https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_games.json.gz

# Steam User Reviews
wget https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz
```

## 🔧 Data Processing Scripts

### Generating Evaluation Datasets

We provide a unified master script `create_datasets.py` to seamlessly build consistent evaluation datasets for all supported sources (MovieLens, Amazon, and Steam). This script filters eligible users, maps metadata (genres, price), extracts chronological histories, and builds structured splits ready for LLM position bias analysis.

```bash
# Generate datasets for all sources
python data/create_datasets.py --dataset all

# Generate datasets for a specific source
python data/create_datasets.py --dataset books
```

**Supported dataset arguments:** `ml-1m`, `books`, `beauty`, `music`, `steam`, `all`.

**Outputs generated per dataset:**
- `test_dataset.json` & `eval_test_dataset.json`: 150 users, 20 candidates (1 ground truth + 19 random negatives).
- `hard_test_dataset.json` & `eval_hard_test_dataset.json`: 100 users, 30 candidates (1 GT + 19 random + 10 genre-matched negatives).
- `probe_dataset.json` & `eval_probe_dataset.json`: 50 users, 100 random unseen candidates (No GT; purely for position probing).

### Loading Data
```python
import pandas as pd
from LLM_debias import LLMPositionBiasAnalyzer

# Load MovieLens data
movielens_data = pd.read_csv('data/ml-1m/processed_ratings.csv')

# Load Amazon data
books_data = pd.read_csv('data/books/ratings_Books.csv')

# Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=movielens_data,
    data_name='movie_lens',
    model='gpt-3.5-turbo',
    backend='openai'
)
```

### Custom Dataset Integration
```python
# For new datasets, modify get_data_columns function
def get_data_columns(data_name: str):
    if data_name == 'your_custom_dataset':
        item_name = 'Title'           # Item column name
        item_metadata = ['Category']  # Metadata columns
        user_metadata = ['Age']       # User columns  
        user_rating = ['Rating']      # Rating column
        return item_name, item_metadata, user_metadata, user_rating
```

## 📊 Dataset Statistics

| Dataset | Users | Items | Interactions | Avg/User | Size |
|---------|-------|-------|--------------|----------|------|
| Beauty | ~2M | ~1M | ~5M | 2.5 | 759MB |
| Books | ~8M | ~6M | ~51M | 6.4 | 16GB |
| MovieLens | 6K | 4K | 1M | 166 | 24MB |
| Music | ~1.5M | ~1M | ~11M | 7.3 | 2.8GB |
| News | ~50K | ~65K | ~0.7M | 14 | 609MB |
| Steam | ~2M | ~32K | ~41M | 20.5 | 4.8GB |

## 🚨 Important Notes

### Data Privacy
- All datasets should be properly anonymized
- Follow data usage agreements and licenses
- Ensure compliance with privacy regulations (GDPR, CCPA)

### Storage Requirements
- **Total Size**: ~24GB for all datasets
- **Recommended**: SSD storage for faster processing
- **Cloud**: Consider cloud storage for large datasets

### Processing Requirements
- **Memory**: 16GB+ RAM recommended for large datasets
- **Processing**: Multi-core CPU for parallel processing
- **Time**: Allow hours for full dataset processing

## 🔍 Data Quality

### Validation Checks
- **Completeness**: No missing required fields
- **Consistency**: Standardized formats across datasets
- **Uniqueness**: No duplicate user-item pairs
- **Validity**: Realistic rating ranges and timestamps

### Known Issues
- **Sparsity**: Some users have very few interactions
- **Bias**: Historical bias present in original data
- **Scale**: Different rating scales across datasets
- **Temporal**: Different time periods covered

## 📚 Citations

### MovieLens
```bibtex
@article{harper2015movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={ACM Transactions on Interactive Intelligent Systems},
  year={2015}
}
```

### Amazon Product Data
```bibtex
@inproceedings{mcauley2015image,
  title={Image-based recommendations on styles and substitutes},
  author={McAuley, Julian and Targett, Christopher and Shi, Qinfeng and Van Den Hengel, Anton},
  booktitle={SIGIR},
  year={2015}
}
```

### MIND News Dataset
```bibtex
@inproceedings{wu2020mind,
  title={MIND: A Large-scale Dataset for News Recommendation},
  author={Wu, Fangzhao and others},
  booktitle={ACL},
  year={2020}
}
```

## 🤝 Contributing

To add new datasets:
1. **Follow naming convention**: `data/{dataset_name}/`
2. **Create README**: Document dataset structure and source
3. **Implement processing**: Add to `get_data_columns()` function
4. **Add tests**: Ensure data loading works correctly
5. **Update documentation**: Add to this README file

---

**⚠️ Note**: Due to large file sizes and licensing restrictions, actual dataset files are not included in this repository. Please download them separately using the instructions above.
