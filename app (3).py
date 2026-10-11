"""
Project 3: Linear Regression - Local Property Price Predictor  (Streamlit version)
Faisalabad house listings (scraped from zameen.com) -> price model using size + location.

Run locally:   streamlit run app.py
Deploy:        push app.py + requirements.txt to GitHub, then deploy on Streamlit Cloud.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SQFT_PER_MARLA = 225  # 1 Marla ~= 225 sq ft
MARLA_PER_KANAL = 20  # 1 Kanal = 20 Marla
CRORE = 1e7

# Raw listings scraped from zameen.com/Houses_Property/Faisalabad-16-{1,2}.html
# Each tuple: (location/society, wider_area, size, size_unit, price, price_unit, zameen_listing_id)
listings = [
    ("Eden Valley", "Faisalabad", 5, "Marla", 2.65, "Crore", "54611735"),
    ("Eden Valley", "Faisalabad", 5, "Marla", 2.6, "Crore", "54611731"),
    ("Eden Valley", "Faisalabad", 7, "Marla", 3.7, "Crore", "54611728"),
    ("Eden Executive", "Eden Gardens", 5, "Marla", 2.6, "Crore", "54611725"),
    ("Eden Executive", "Eden Gardens", 5, "Marla", 2.3, "Crore", "54614817"),
    ("Eden Valley", "Faisalabad", 5, "Marla", 2.65, "Crore", "54622884"),
    ("Abdullah Gardens", "East Canal Road", 20, "Marla", 14.0, "Crore", "54496692"),
    ("Eden Orchard", "Faisalabad", 5, "Marla", 2.7, "Crore", "54575043"),
    ("Al Najaf Colony", "Faisalabad", 2.5, "Marla", 1.5, "Crore", "54615181"),
    ("Divine Enclave", "Canal Road", 13, "Marla", 6.5, "Crore", "54571147"),
    ("Khayaban Colony 2", "Faisalabad", 22, "Marla", 5.6, "Crore", "53657911"),
    ("Model City 1", "Canal Road", 7, "Marla", 3.5, "Crore", "54306077"),
    ("Model City 1", "Canal Road", 3, "Marla", 1.28, "Crore", "54571133"),
    ("Eden Executive", "Eden Gardens", 5, "Marla", 2.1, "Crore", "54571121"),
    ("Eden Valley", "Faisalabad", 7, "Marla", 3.6, "Crore", "53493139"),
    ("Eden Orchard", "Faisalabad", 11, "Marla", 6.0, "Crore", "54545939"),
    ("Canal Road", "Faisalabad", 7, "Marla", 2.5, "Crore", "53876022"),
    ("Model City 1", "Canal Road", 7, "Marla", 2.7, "Crore", "54622854"),
    ("Model City 1", "Canal Road", 3, "Marla", 1.35, "Crore", "54622697"),
    ("Kohinoor City", "Faisalabad", 16, "Marla", 8.5, "Crore", "54070243"),
    ("Citi Housing", "Faisalabad", 10, "Marla", 3.35, "Crore", "54588956"),
    ("Citi Housing", "Faisalabad", 10, "Marla", 3.0, "Crore", "54588985"),
    ("Citi Housing", "Faisalabad", 5, "Marla", 1.35, "Crore", "54588986"),
    ("Citi Housing Sargodha Road", "Citi Housing", 42, "Marla", 9.9, "Crore", "54588988"),
    ("Citi Housing Sargodha Road", "Citi Housing", 20, "Marla", 7.5, "Crore", "54588989"),
    ("Eden Valley", "Faisalabad", 7, "Marla", 4.25, "Crore", "54497801"),
    ("Eden Valley", "Faisalabad", 5, "Marla", 2.7, "Crore", "54497803"),
    ("Wapda City - Block M", "Wapda City", 10, "Marla", 4.75, "Crore", "54581736"),
    ("Eden Orchard", "Faisalabad", 5, "Marla", 2.4, "Crore", "54605383"),
    ("Divine Enclave", "Canal Road", 7, "Marla", 4.25, "Crore", "54600470"),
    ("Wapda City", "Faisalabad", 5, "Marla", 2.55, "Crore", "54428116"),
    ("Eden Orchard", "Faisalabad", 5, "Marla", 2.6, "Crore", "54494648"),
    ("Eden Orchard", "Faisalabad", 11, "Marla", 6.0, "Crore", "54604116"),
    ("Eden Orchard Block Y", "Eden Orchard", 5, "Marla", 2.8, "Crore", "54350345"),
    ("Jaranwala Road", "Faisalabad", 2.2, "Marla", 0.60, "Crore", "54518182"),
    ("Eden Valley", "Faisalabad", 6, "Marla", 3.6, "Crore", "54318366"),
    ("Eden Executive", "Eden Gardens", 7, "Marla", 2.75, "Crore", "54554089"),
    ("Eden Orchard", "Faisalabad", 5, "Marla", 2.25, "Crore", "54622431"),
    ("Al Najaf Colony", "Faisalabad", 10, "Marla", 5.0, "Crore", "54615268"),
    ("Peoples Colony - Block B", "Peoples Colony No 1", 7.5, "Marla", 5.0, "Crore", "54577386"),
    ("Eden Orchard", "Faisalabad", 5, "Marla", 2.6, "Crore", "54444126"),
    ("Satiana Road", "Faisalabad", 8, "Marla", 1.7, "Crore", "53055130"),
    ("Citi Housing Sargodha Road", "Citi Housing", 5, "Marla", 1.55, "Crore", "54588990"),
    ("Citi Housing Sargodha Road", "Citi Housing", 20, "Marla", 10.0, "Crore", "54588991"),
    ("Divine Enclave", "Canal Road", 7, "Marla", 4.25, "Crore", "54619730"),
    ("Eden Executive", "Eden Gardens", 5, "Marla", 2.3, "Crore", "54619728"),
    ("Eden Executive", "Eden Gardens", 2.7, "Marla", 1.25, "Crore", "54619727"),
    ("Model City 1", "Canal Road", 5, "Marla", 1.85, "Crore", "54619726"),
    ("Abdullah Gardens", "East Canal Road", 7, "Marla", 4.0, "Crore", "54619723"),
    ("Eden Executive", "Eden Gardens", 7, "Marla", 3.5, "Crore", "54595655"),
]


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------
@st.cache_data
def load_listings():
    rows = []
    for loc, area, size_val, size_unit, price_val, price_unit, listing_id in listings:
        marla = size_val * MARLA_PER_KANAL if size_unit == "Kanal" else size_val
        rows.append(
            {
                "Location": loc,
                "Wider_Area": area,
                "Size_Marla": marla,
                "Size_Sqft": marla * SQFT_PER_MARLA,
                "Price_PKR": price_val * CRORE if price_unit == "Crore" else price_val * 1e5,
                "Zameen_Listing_ID": listing_id,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def train(min_listings, test_size, seed):
    df = load_listings().copy()
    df["Price_per_Marla"] = df["Price_PKR"] / df["Size_Marla"]

    counts = df["Location"].value_counts()
    common = counts[counts >= min_listings].index.tolist()
    df["Location_Grouped"] = df["Location"].where(df["Location"].isin(common), "Other")

    X = df[["Size_Sqft", "Location_Grouped"]]
    y = df["Price_PKR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    preprocessor = ColumnTransformer(
        [("location_ohe", OneHotEncoder(handle_unknown="ignore"), ["Location_Grouped"])],
        remainder="passthrough",
    )
    model = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    pred_train, pred_test = model.predict(X_train), model.predict(X_test)
    metrics = {
        "train_mse": mean_squared_error(y_train, pred_train),
        "test_mse": mean_squared_error(y_test, pred_test),
        "train_r2": r2_score(y_train, pred_train),
        "test_r2": r2_score(y_test, pred_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Coefficients: one effect per location category + PKR per sq ft
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = pd.DataFrame(
        {"Feature": [n.replace("location_ohe__Location_Grouped_", "Location: ").replace("remainder__", "") for n in names],
         "Coefficient (PKR)": model.named_steps["regressor"].coef_}
    )
    test_df = pd.DataFrame(
        {"Actual (Crore)": y_test.to_numpy() / CRORE, "Predicted (Crore)": pred_test / CRORE,
         "Location": X_test["Location_Grouped"].to_numpy(), "Size (Marla)": X_test["Size_Sqft"].to_numpy() / SQFT_PER_MARLA}
    )
    return df, common, model, metrics, coefs, test_df


def crore(x):
    return f"PKR {x / CRORE:,.2f} Crore"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Faisalabad Property Price Predictor")
st.caption("Linear regression on 50 real house listings (zameen.com): price from size + society.")

with st.sidebar:
    st.header("Model settings")
    min_listings = st.slider("Min listings for a society to be its own category", 2, 6, 3)
    test_size = st.slider("Test set share", 0.1, 0.4, 0.2, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.caption("Societies with fewer listings are grouped into **Other**.")

df, common, model, metrics, coefs, test_df = train(min_listings, test_size, int(seed))

tab_data, tab_explore, tab_model, tab_predict = st.tabs(
    ["📦 Data", "📊 Explore", "🧮 Model", "🎯 Predict a price"]
)

# ----- Data -----
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Listings", len(df))
    c2.metric("Missing values", int(df.isnull().sum().sum()))
    c3.metric(
        "Duplicate (society, size, price)",
        int(df.duplicated(subset=["Location", "Size_Marla", "Price_PKR"]).sum()),
    )
    st.dataframe(
        df[["Location", "Wider_Area", "Size_Marla", "Size_Sqft", "Price_PKR", "Zameen_Listing_ID"]],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Summary statistics")
    st.dataframe(df[["Size_Marla", "Size_Sqft", "Price_PKR"]].describe().round(1), width="stretch")
    st.caption(
        "Units: 1 Kanal = 20 Marla, 1 Marla ≈ 225 sq ft, 1 Crore = 10,000,000 PKR. "
        "Listings with the same society, size and price are kept (they may be different houses)."
    )

# ----- Explore -----
with tab_explore:
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution of listed prices")
        counts_, edges = np.histogram(df["Price_PKR"] / CRORE, bins=15)
        hist = pd.DataFrame({"Price (Crore)": [f"{e:.1f}" for e in edges[:-1]], "Listings": counts_})
        st.bar_chart(hist, x="Price (Crore)", y="Listings")
    with right:
        st.subheader("Price vs. size")
        st.scatter_chart(
            df.assign(**{"Price (Crore)": df["Price_PKR"] / CRORE}),
            x="Size_Marla", y="Price (Crore)", color="Location_Grouped",
        )
    left, right = st.columns(2)
    with left:
        st.subheader("Listings per society (top 8)")
        st.bar_chart(df["Location"].value_counts().head(8))
    with right:
        st.subheader("Average price per Marla (million PKR)")
        ppm = (df.groupby("Location")["Price_per_Marla"].mean().sort_values(ascending=False).head(10) / 1e6)
        st.bar_chart(ppm)
    st.info(
        f"Kept as their own category (≥ {min_listings} listings): " + ", ".join(common)
        + ". Everything else → **Other**."
    )

# ----- Model -----
with tab_model:
    st.write(f"Trained on **{metrics['n_train']}** listings, tested on **{metrics['n_test']}**.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train R²", f"{metrics['train_r2']:.3f}")
    m2.metric("Test R²", f"{metrics['test_r2']:.3f}")
    m3.metric("Train RMSE", crore(np.sqrt(metrics["train_mse"])))
    m4.metric("Test RMSE", crore(np.sqrt(metrics["test_mse"])))
    st.caption("RMSE is the typical size of the prediction error. With only a few test rows, treat these numbers as rough.")

    lo = float(min(test_df["Actual (Crore)"].min(), test_df["Predicted (Crore)"].min()))
    hi = float(max(test_df["Actual (Crore)"].max(), test_df["Predicted (Crore)"].max()))
    points = (
        alt.Chart(test_df)
        .mark_circle(size=90)
        .encode(x="Actual (Crore)", y="Predicted (Crore)", tooltip=list(test_df.columns))
    )
    line = (
        alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]}))
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x", y="y")
    )
    st.subheader("Actual vs. predicted (test set)")
    st.altair_chart(points + line, width="stretch")
    st.caption("Points on the red dashed line would be perfect predictions.")

    st.subheader("What the model learned")
    st.dataframe(coefs.round(0), width="stretch", hide_index=True)
    st.caption(
        "Size_Sqft = PKR added per extra square foot. Location rows are the price premium or discount "
        "of that society compared with the others, at the same size."
    )

# ----- Predict -----
with tab_predict:
    st.subheader("Estimate a house price")
    c1, c2, c3 = st.columns(3)
    unit = c1.selectbox("Size unit", ["Marla", "Kanal", "Sq ft"])
    default_size = {"Marla": 5.0, "Kanal": 1.0, "Sq ft": 1125.0}[unit]
    size_val = c2.number_input(f"Size ({unit})", min_value=0.1, value=default_size, step=0.5)
    society = c3.selectbox("Society", sorted(common) + ["Other / not listed"])

    marla = {"Marla": size_val, "Kanal": size_val * MARLA_PER_KANAL, "Sq ft": size_val / SQFT_PER_MARLA}[unit]
    sqft = marla * SQFT_PER_MARLA
    group = "Other" if society == "Other / not listed" else society

    pred = float(model.predict(pd.DataFrame([{"Size_Sqft": sqft, "Location_Grouped": group}]))[0])

    st.metric("Estimated price", crore(pred) if pred > 0 else "Not reliable")
    st.write(f"≈ PKR {pred:,.0f}" if pred > 0 else "")
    st.caption(f"{marla:,.1f} Marla = {sqft:,.0f} sq ft in **{group}**")

    # Honest warnings
    lo_m, hi_m = df["Size_Marla"].min(), df["Size_Marla"].max()
    if marla < lo_m or marla > hi_m:
        st.warning(f"Outside the training range ({lo_m:g}–{hi_m:g} Marla). The model is extrapolating, so treat this as a rough guess.")
    group_max = df.loc[df["Location_Grouped"] == group, "Size_Marla"].max()
    if marla > group_max * 1.5:
        st.warning(f"The largest {group} listing in the data is {group_max:g} Marla, so this size is far from anything the model saw there.")
    if group == "Other":
        st.info("'Other' mixes many societies, so the estimate is less specific.")
    if pred <= 0:
        st.error("A straight-line model can go negative for very small plots. Don't use this number.")

    st.divider()
    st.subheader("Example scenarios (from the notebook)")
    other_for_wapda = "Wapda City" if "Wapda City" in common else "Other"
    examples = pd.DataFrame(
        [
            {"Scenario": "5 Marla in Eden Valley (well represented)", "Size_Sqft": 5 * SQFT_PER_MARLA, "Location_Grouped": "Eden Valley"},
            {"Scenario": "10 Marla in Wapda City (sparse society)", "Size_Sqft": 10 * SQFT_PER_MARLA, "Location_Grouped": other_for_wapda},
            {"Scenario": "30 Marla in Eden Orchard (size extrapolation)", "Size_Sqft": 30 * SQFT_PER_MARLA, "Location_Grouped": "Eden Orchard"},
        ]
    )
    valid = examples["Location_Grouped"].isin(df["Location_Grouped"].unique())
    examples = examples[valid].copy()
    if not examples.empty:
        examples["Predicted (Crore)"] = (model.predict(examples[["Size_Sqft", "Location_Grouped"]]) / CRORE).round(2)
        st.dataframe(examples[["Scenario", "Predicted (Crore)"]], width="stretch", hide_index=True)
