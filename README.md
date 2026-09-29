# 🏠 Local Property Price Predictor: Linear Regression

A machine learning project that predicts residential house prices in **Faisalabad, Pakistan** using multiple linear regression. The model is trained on real property listings from Zameen.com and uses plot size and housing society (location) as predictors.

---

## 📌 Overview

Property prices in Pakistani cities are typically quoted in **Marla** (plot size) and **Crore** (PKR 10 million), and vary sharply between housing societies. This project builds an end-to-end regression workflow, from data collection and cleaning to modeling, evaluation, and prediction, that captures both of those effects.

## 🎯 Objectives

- Assemble a dataset from **50 real Faisalabad house listings**
- Standardize local units (Marla → sq ft, Crore/Lakh → PKR)
- Explore how price relates to size and location
- Build a **scikit-learn Pipeline** that combines one-hot encoding with linear regression
- Evaluate the model on unseen data using **MSE, RMSE, and R²**
- Test predictions on hypothetical properties, including edge cases

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas / NumPy | Data preparation and numerical work |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Train/test split, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `LinearRegression`, metrics |
| Jupyter / Google Colab | Development environment |

## 📂 Dataset

50 active house listings collected from Zameen.com (Faisalabad), stored directly in the notebook. Each record contains:

| Field | Description |
|-------|-------------|
| `Location` | Housing society (e.g., Eden Valley, Citi Housing, Model City 1) |
| `Wider_Area` | Broader area (e.g., Canal Road, Eden Gardens) |
| `Size_Marla` | Plot size in Marla |
| `Size_Sqft` | Plot size in square feet |
| `Price_PKR` | Listed price in Pakistani Rupees |
| `Zameen_Listing_ID` | Source listing ID for traceability |

**Unit conversions:** 1 Kanal = 20 Marla, 1 Marla ≈ 225 sq ft, 1 Crore = 10,000,000 PKR, 1 Lakh = 100,000 PKR.

**Data summary:**

| Statistic | Size (Marla) | Price (PKR) |
|-----------|-------------|-------------|
| Mean | 8.3 | 37.6 M |
| Median | 7.0 | 27.3 M |
| Min | 2.2 | 6.0 M |
| Max | 42 | 140 M |

The data has no missing values or invalid entries (zero or negative price or size).

## ⚙️ Methodology

1. **Data preparation**: Convert local units into sq ft and PKR, then build a Pandas DataFrame.
2. **Data quality checks**: Check for missing values, invalid entries, and duplicates.
3. **Exploratory analysis**: Plot the price distribution, price vs. size, and listings per society. Compute average price per Marla by society.
4. **Feature engineering**: Keep societies with at least 3 listings as their own category and group the rest into `Other`. This reduces noise from rarely seen locations.
5. **Modeling**: Build a `Pipeline` with a `ColumnTransformer` (one-hot encoding of location, with size passed through) followed by `LinearRegression`.
6. **Evaluation**: Use an 80/20 train/test split (`random_state=42`) and report MSE, RMSE, and R².
7. **Prediction**: Estimate prices for three hypothetical properties.

## 📊 Results

| Metric | Train (40 rows) | Test (10 rows) |
|--------|-----------------|----------------|
| RMSE | PKR 14.6 M | PKR 10.0 M |
| R² | 0.700 | 0.734 |

The model explains roughly **70–73% of the variance** in listed prices, with test performance in line with training performance.

**Sample predictions:**

| Scenario | Predicted Price |
|----------|-----------------|
| 5 Marla in Eden Valley (well-represented society) | ≈ PKR 2.87 Crore |
| 10 Marla in a sparse society (grouped as "Other") | ≈ PKR 4.39 Crore |
| 30 Marla in Eden Orchard (size extrapolation) | ≈ PKR 10.53 Crore |

## 🔍 Key Insights

- **Size is the dominant driver** of price, but society matters. Average price per Marla ranges from about PKR 4.75 M to 6.7 M across the top societies.
- Grouping rare societies into `Other` keeps the model stable when some locations have very few listings.
- Using `OneHotEncoder(handle_unknown="ignore")` lets the pipeline handle societies it has never seen without failing.
- The 30 Marla example is an extrapolation, so treat it with caution.

## ⚠️ Limitations

- **Small sample**: With 50 listings and only 10 test rows, the metrics are noisy and can change noticeably with a different split.
- **Asking prices, not sale prices**: Listings show what sellers want, which can differ from final transaction values.
- **Limited features**: The model ignores bedrooms, construction quality, corner plots, and plot condition.
- **Linear assumption**: Real prices may scale non-linearly with size, and a few large plots have high leverage.
- **Duplicate-like rows**: Five rows share the same location, size, and price, which may be repeated listings.
- **Point-in-time data**: Prices reflect the market when the listings were collected.

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/property-price-predictor.git
   cd property-price-predictor
   ```
2. Open `Project_3___Linear_Regression_Local_Property_Price_Predictor.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/).
3. Run all cells from top to bottom. No external data file is needed because the listings are embedded in the notebook.

## 📁 Project Structure

```
├── Project_3___Linear_Regression_Local_Property_Price_Predictor.ipynb   # Full analysis and model
└── README.md
```

## 🔮 Future Improvements

- Collect a larger dataset across multiple pages, cities, and time periods
- Add features such as bedrooms, bathrooms, corner plot, and age of construction
- Try log-transforming price or size to handle skewness
- Compare with Ridge, Lasso, Random Forest, and Gradient Boosting models
- Use cross-validation for more reliable performance estimates
- Deploy as a simple web app (Streamlit or Flask) for interactive price estimates

## 👤 Author

Alia Maryam
[GitHub](https://github.com/Alia Maryam) · [LinkedIn](www.linkedin.com/in/ alia-maryam-a59a47355)
