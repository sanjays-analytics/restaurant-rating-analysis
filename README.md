# Restaurant Rating Analysis & Segmentation

Independent analysis of a large-scale restaurant dataset (branded **FoodieBay** for this project; the underlying data follows the structure of Zomato's restaurant listings), predicting restaurant ratings and segmenting restaurants into meaningful groups using supervised and unsupervised machine learning.

## Status: Complete

## Business framing

Analysed using the Business Analysis Core Concept Model (BACCM):

- **Context**: FoodieBay operates a digital platform listing partner restaurants, their features, and customer reviews.
- **Need**: growing competition among restaurants makes understanding what actually drives ratings a genuine business question, not just an academic one.
- **Stakeholders**: restaurant owners and management, delivery platforms, and the customers who rely on ratings to choose where to eat.
- **Value**: identifying which factors most influence ratings lets restaurants act on the ones they can control.
- **Solution**: a predictive model for ratings, plus a clustering model to segment restaurants by characteristics.
- **Change**: a working prediction pipeline that could plausibly be integrated into the platform itself.

## Dataset

40,130 restaurant listings across India, 17 columns, covering location, cuisine, table booking and online ordering availability, average cost for two, review ranking, vote count, and overall rating (`rate`).

Six columns had meaningful missing data: `dish_liked` (22,779 rows), `rate` (8,336), `ave_review_ranking` (6,379), `phone` (884), `ave_cost_for_two` (240), and `cuisines` (18).

## Tools

Python, Pandas, NumPy, Scikit-learn, Keras (TensorFlow), Matplotlib, Seaborn

## Methodology

**1. Data preparation.** Missing numeric values (`ave_cost_for_two`, `ave_review_ranking`, `rate`) were imputed using median or mean depending on each variable's skew, checked visually via histograms before deciding which measure to use. High-cardinality text fields (`dish_liked`, `cuisines`, `phone`) were deliberately left unimputed rather than filled with a most-common value, since a single restaurant's specific cuisine or signature dish can't be reasonably guessed from the rest of the dataset.

**2. Exploratory Data Analysis.** Univariate, bivariate, and multivariate analysis (including a full correlation heatmap) to understand each feature's distribution and its relationship to `rate`, used to guide feature selection before modelling.

**3. Supervised learning: rating prediction.** Six distinct techniques were trained and compared, treating rating prediction as a regression problem:

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.390 | 0.302 |
| Decision Tree | 0.572 | 0.252 |
| KNN (k=5) | 0.737 | 0.198 |
| KNN (k=2, tuned) | 0.767 | 0.186 |
| ANN (Scikit MLPRegressor) | 0.536 | 0.198 |
| ANN (Keras neural network) | 0.610 | 0.160 |
| Random Forest | 0.843 | 0.153 |
| AdaBoost | 0.439 | 0.289 |
| Gradient Boosting | 0.633 | 0.234 |
| **Stacking Ensemble (final)** | **0.843** | **0.153** |

The final model stacks Random Forest, AdaBoost, and Gradient Boosting regressors, passed through a Linear Regression meta-estimator. It explains 84.3% of the variance in ratings with a mean absolute error of 0.073, the strongest result of any approach tested, and was selected over the next-best model (tuned KNN) on that basis.

**4. Unsupervised learning: restaurant segmentation.** K-Means clustering was used to group restaurants by shared characteristics. Rather than picking a cluster count arbitrarily, the optimal *k* was cross-checked across three independent methods:

| Method | Suggested k |
|---|---|
| Elbow Method | 2 |
| Davies-Bouldin Index | 3 |
| Silhouette Score | 3 |

*k* = 3 was selected on the majority result. The resulting segments:

- **Cluster 0**: no table booking, lower review ranking, moderate cost, moderate popularity
- **Cluster 1**: no table booking, higher review ranking, moderate cost, moderate popularity
- **Cluster 2**: table booking available, higher review ranking, higher cost, high popularity

Cluster 2 reads as the clear "premium" segment: table booking, higher prices, and higher popularity all cluster together.

## Key findings

- **Table booking has a real association with rating**: only 10.2% of restaurants offer it, but restaurants that do average a 4.10 rating versus 3.60 for those that don't.
- **Online ordering is widespread (61.2%) but doesn't move ratings much**, unlike table booking, it doesn't appear to be a strong differentiator on its own.
- **Price and rating are connected but not linearly simple**: restaurants rated 3.0-4.0 mostly sit in the 200-1000 INR range for two, while restaurants rated above 4.0 skew toward higher price points, consistent with a premium-experience effect rather than price alone driving satisfaction.
- **Casual Dining with Bar service rates highest on average (4.06)**, while Quick Bites is both the most common restaurant type and the cheapest (around 300 INR for two).
- **Review ranking correlates with rating in a clean linear pattern**, unsurprising, but a useful confirmation that the two metrics are measuring related things rather than redundant ones.

## Data quality notes

- Missing-value treatment was chosen per-variable based on actual distribution shape rather than defaulting to one method across the board.
- High-cardinality categorical fields were deliberately left missing rather than imputed, since a plausible-looking fill value (e.g. the most common cuisine) could actively mislead downstream analysis for an individual restaurant.

## Limitations and future work

- The dataset doesn't include customer demographic data, which would meaningfully deepen the segmentation and prediction work if it became available.
- Ratings reflect a snapshot in time rather than a trend; a time-series view of how ratings evolve after a restaurant adds table booking or online ordering would strengthen the causal read on those two features.
- Cluster interpretation here is based on descriptive statistics per group; a follow-up profiling pass naming each segment for a non-technical business audience would make the clustering more directly actionable.

## Repository structure

```
restaurant-rating-analysis/
├── data/
│   ├── FoodieBay.csv
│   └── FoodieBay_metadata.csv
├── restaurant_rating_analysis.ipynb
└── README.md
```
