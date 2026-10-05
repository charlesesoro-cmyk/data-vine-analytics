# DataVine Analytics: Machine Learning Lab

A machine learning lab for **DataVine Analytics** covering three tasks in one Jupyter notebook: classification, recommendation, and clustering. Completed as a summative lab for the Moringa School Data Science program.

**Notebook:** [`data_vine_analytics.ipynb`](data_vine_analytics.ipynb)

## Projects

| # | Project | Dataset | Techniques |
|---|---------|---------|------------|
| 1 | Wine classification | `wine.csv` (178 samples, 13 chemical features, 3 classes) | StandardScaler, PCA (95% variance), k-NN, GridSearchCV |
| 2 | Agricultural feed recommendation | `chickwts.csv` (70 rows after de-duplication, 6 feed types) | Group aggregation, PCA, cosine similarity |
| 3 | Regional crime pattern analysis | `USArrests.csv` (50 US states) | Correlation-based feature selection, PCA, K-Means, Gaussian Mixture Model |

## Workflow

Data preparation → exploratory analysis → preprocessing → dimensionality reduction → model building → hyperparameter tuning → evaluation → visualization → interpretation

## Key Results

### 1. Wine classification
- PCA reduced 13 features to **10 components**, retaining **96.2%** of the variance.
- Best k-NN parameters from 5-fold GridSearchCV: `k = 7`, Euclidean distance, uniform weights.
- Cross-validation accuracy: **97.2%**. Test accuracy: **100%** (36 test samples).

### 2. Feed recommendation
- Average chicken weight per feed: sunflower (329) > casein (324) > meatmeal (277) > soybean (246) > linseed (219) > horsebean (160).
- Feeds are standardized, reduced to 1 principal component, and compared by cosine similarity. A `recommend_feeds()` function returns the top matches for any feed.
- **Limitation:** the dataset has only one performance variable, so cosine similarity in one dimension collapses to +1 or -1 (feeds above or below the mean). Treat this as a prototype, not a production recommender.

### 3. Crime clustering
- The three variables most correlated with the others were selected: **Assault, Rape, Murder** (UrbanPop was dropped).
- Two principal components explain **93.9%** of the variance.
- **K-Means:** 4 clusters chosen from the elbow plot, silhouette score **0.464** (cluster sizes 14 / 12 / 17 / 7).
- **GMM:** BIC selected **2 components** (BIC = 293.28), with sizes 31 / 19 and per-state membership probabilities.
- K-Means and GMM assignments are compared side by side for every state.

## Repository Structure

```
data-vine-analytics/
├── data_vine_analytics.ipynb   # Full analysis
├── wine.csv                    # Wine dataset
├── chickwts.csv                # Chick weight / feed dataset
├── USArrests.csv               # US state crime statistics
└── README.md
```

> The notebook loads the three CSV files from its own folder, so keep them together.

## Tools & Libraries

Python 3, Jupyter, pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## How to Run

```bash
git clone https://github.com/charlesesoro-cmyk/data-vine-analytics.git
cd data-vine-analytics
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook data_vine_analytics.ipynb
```

Then run all cells from top to bottom.

## Author

**Charles Esoro**
GitHub: [@charlesesoro-cmyk](https://github.com/charlesesoro-cmyk)
