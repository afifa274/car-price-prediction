"""
app.py
------
Streamlit frontend for Car Price Prediction.
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (minimal) ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .big-price {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a56db;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("cardekho.csv")
    for col in ["mileage(km/ltr/kg)", "engine", "max_power"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
            )
    df["car_age"] = 2024 - df["year"]
    df.dropna(subset=["selling_price", "mileage(km/ltr/kg)",
                       "engine", "max_power", "seats"], inplace=True)
    return df

try:
    artefact = load_model()
    pipeline = artefact["pipeline"]
    meta     = artefact["meta"]
    model_ready = True
except FileNotFoundError:
    model_ready = False

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
             use_column_width=True, caption="")
    st.title("🚗 Car Price Predictor")
    st.markdown("---")
    st.markdown("**Dataset:** CarDekho")
    st.markdown(f"**Records:** {len(df):,}")
    if model_ready:
        st.markdown(f"**Model R²:** {meta['r2']}")
        st.markdown(f"**Model MAE:** ₹{meta['mae']:,.0f}")
    st.markdown("---")
    st.markdown("Built with **Streamlit** + **scikit-learn**")

# ── Main title ────────────────────────────────────────────────────────────────
st.title("🚗 Car Price Prediction")
st.markdown("Predict the resale price of any used car using a trained **Random Forest** model on the CarDekho dataset.")

if not model_ready:
    st.error("⚠️ model.pkl not found. Please run `python train_model.py` first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_explore, tab_model = st.tabs(
    ["🔮 Predict Price", "📊 Data Explorer", "🧠 Model Insights"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Enter Car Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.slider(
            "Manufacturing Year",
            min_value=meta["min_year"],
            max_value=meta["max_year"],
            value=2018,
        )
        car_age = 2024 - year

        km_driven = st.number_input(
            "Kilometres Driven",
            min_value=0, max_value=500_000,
            value=40_000, step=1000,
        )
        fuel = st.selectbox("Fuel Type", meta["unique_fuels"])

    with col2:
        transmission = st.selectbox("Transmission", meta["unique_transmissions"])
        seller_type  = st.selectbox("Seller Type",  meta["unique_seller_types"])
        owner        = st.selectbox("Owner",         meta["unique_owners"])

    with col3:
        mileage   = st.number_input("Mileage (km/ltr or km/kg)", min_value=5.0,  max_value=40.0,  value=18.0, step=0.1)
        engine    = st.number_input("Engine Displacement (cc)",   min_value=500,  max_value=5000,  value=1200, step=50)
        max_power = st.number_input("Max Power (bhp)",            min_value=30.0, max_value=600.0, value=80.0, step=1.0)
        seats     = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

    st.markdown("---")
    predict_btn = st.button("💰 Predict Selling Price", type="primary", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "car_age":              car_age,
            "km_driven":            km_driven,
            "mileage(km/ltr/kg)":   mileage,
            "engine":               float(engine),
            "max_power":            max_power,
            "seats":                float(seats),
            "fuel":                 fuel,
            "seller_type":          seller_type,
            "transmission":         transmission,
            "owner":                owner,
        }])

        predicted_price = pipeline.predict(input_data)[0]

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.markdown(
                f"""<div class="metric-card">
                    <p style="margin:0;color:#555;font-size:1rem;">Estimated Resale Price</p>
                    <p class="big-price">₹{predicted_price:,.0f}</p>
                    <p style="margin:0;color:#888;font-size:0.85rem;">
                        Confidence range: ₹{max(0, predicted_price - meta['mae']):,.0f}
                        – ₹{predicted_price + meta['mae']:,.0f}
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

        # Feature importance mini-bar for this prediction
        st.markdown("#### Feature Values Used")
        feat_df = pd.DataFrame({
            "Feature": ["Car Age", "KM Driven", "Mileage", "Engine", "Max Power", "Seats",
                         "Fuel", "Seller Type", "Transmission", "Owner"],
            "Value": [car_age, km_driven, mileage, engine, max_power, seats,
                      fuel, seller_type, transmission, owner],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.subheader("Explore the CarDekho Dataset")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(df):,}")
    k2.metric("Avg Selling Price", f"₹{df['selling_price'].mean():,.0f}")
    k3.metric("Median Selling Price", f"₹{df['selling_price'].median():,.0f}")
    k4.metric("Unique Car Names", str(df["name"].nunique()))

    st.markdown("---")

    ecol1, ecol2 = st.columns(2)

    with ecol1:
        # Price distribution
        fig_hist = px.histogram(
            df, x="selling_price", nbins=60,
            title="Selling Price Distribution",
            labels={"selling_price": "Selling Price (₹)"},
            color_discrete_sequence=["#1a56db"],
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Fuel type vs price
        fig_box = px.box(
            df, x="fuel", y="selling_price",
            title="Price by Fuel Type",
            labels={"selling_price": "Selling Price (₹)", "fuel": "Fuel Type"},
            color="fuel",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with ecol2:
        # Scatter: KM driven vs Price
        fig_scatter = px.scatter(
            df.sample(min(2000, len(df)), random_state=1),
            x="km_driven", y="selling_price",
            color="fuel", opacity=0.6,
            title="KM Driven vs Selling Price",
            labels={"km_driven": "KM Driven", "selling_price": "Selling Price (₹)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Count by transmission
        trans_counts = df["transmission"].value_counts().reset_index()
        trans_counts.columns = ["Transmission", "Count"]
        fig_pie = px.pie(
            trans_counts, names="Transmission", values="Count",
            title="Transmission Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Year vs average price trend
    year_price = df.groupby("year")["selling_price"].mean().reset_index()
    fig_trend = px.line(
        year_price, x="year", y="selling_price",
        title="Average Selling Price by Manufacturing Year",
        labels={"year": "Year", "selling_price": "Avg Selling Price (₹)"},
        markers=True,
        color_discrete_sequence=["#1a56db"],
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    n_rows = st.slider("Rows to display", 5, 100, 20)
    st.dataframe(df.head(n_rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.subheader("Model Performance & Feature Importance")

    m1, m2 = st.columns(2)
    m1.metric("R² Score (test set)",           f"{meta['r2']}")
    m2.metric("Mean Absolute Error (test set)", f"₹{meta['mae']:,.0f}")

    st.markdown("---")

    # Feature importances from the RF model
    rf_model    = pipeline.named_steps["model"]
    num_cols    = meta["numeric_cols"]
    cat_cols    = meta["categorical_cols"]
    feat_names  = num_cols + cat_cols

    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    feat_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        title="Random Forest Feature Importances",
        color="Importance",
        color_continuous_scale="Blues",
        labels={"Importance": "Importance Score"},
    )
    fig_fi.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # Seller type vs price
    seller_avg = df.groupby("seller_type")["selling_price"].mean().reset_index()
    fig_seller = px.bar(
        seller_avg, x="seller_type", y="selling_price",
        title="Average Selling Price by Seller Type",
        color="seller_type",
        labels={"selling_price": "Avg Selling Price (₹)", "seller_type": "Seller Type"},
    )
    st.plotly_chart(fig_seller, use_container_width=True)

    # Owner type vs price
    owner_avg = df.groupby("owner")["selling_price"].mean().reset_index()
    fig_owner = px.bar(
        owner_avg, x="owner", y="selling_price",
        title="Average Selling Price by Owner Type",
        color="owner",
        labels={"selling_price": "Avg Selling Price (₹)", "owner": "Owner Type"},
    )
    st.plotly_chart(fig_owner, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Model Details**
    | Parameter | Value |
    |---|---|
    | Algorithm | Random Forest Regressor |
    | n_estimators | 200 |
    | max_depth | 15 |
    | Numerical scaling | StandardScaler |
    | Categorical encoding | OrdinalEncoder |
    | Train/Test split | 80 / 20 |
    """)
