# ============================================================
# STUDENT PERFORMANCE ANALYZER
# Streamlit Dashboard - Dark/Light Mode Safe
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)

from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }


    /* ==============================
       MAIN HEADINGS
       ============================== */

    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #2563eb !important;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b !important;
        font-size: 16px;
        margin-bottom: 25px;
    }


    /* ==============================
       KPI CARDS
       ============================== */

    .kpi-card {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dbe3ec;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }

    .kpi-title {
        color: #64748b !important;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #0f172a !important;
        font-size: 28px;
        font-weight: 700;
    }


    /* ==============================
       PREDICTION CARD
       ============================== */

    .prediction-card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #dbe3ec;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-card * {
        color: #0f172a !important;
    }


    /* ==============================
       INPUT LABELS
       ============================== */

    label {
        font-weight: 500;
    }


    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PLOTLY THEME FUNCTION
# ============================================================

def style_chart(fig, height=450):

    fig.update_layout(

        height=height,

        plot_bgcolor="#ffffff",

        paper_bgcolor="#ffffff",

        font=dict(
            color="#0f172a"
        ),

        title_font=dict(
            color="#0f172a"
        ),

        xaxis=dict(
            title_font=dict(
                color="#0f172a"
            ),

            tickfont=dict(
                color="#334155"
            ),

            gridcolor="#e2e8f0",

            linecolor="#cbd5e1"
        ),

        yaxis=dict(
            title_font=dict(
                color="#0f172a"
            ),

            tickfont=dict(
                color="#334155"
            ),

            gridcolor="#e2e8f0",

            linecolor="#cbd5e1"
        ),

        legend=dict(
            font=dict(
                color="#0f172a"
            )
        )

    )

    return fig



# ============================================================
# LOAD DATA - FIXED PATH
# ============================================================

from pathlib import Path

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "Student_Performance_Analyzed.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

except Exception as e:
    st.error(f"Dataset could not be loaded: {e}")
    st.write("Expected location:", str(DATA_PATH))
    st.stop()



# ============================================================
# FEATURE ENGINEERING
# ============================================================

if "Study Time" not in df.columns:

    df["Study Time"] = df["Study_Hours_Per_Day"]


if "Total Grade" not in df.columns:

    df["Total Grade"] = (

        df["Math_Score"]

        + df["Science_Score"]

        + df["English_Score"]

    )


if "Failures" not in df.columns:

    df["Failures"] = (

        (df["Math_Score"] < 40).astype(int)

        + (df["Science_Score"] < 40).astype(int)

        + (df["English_Score"] < 40).astype(int)

    )


if "Absences" not in df.columns:

    df["Absences"] = (

        100 - df["Attendance_Percentage"]

    )


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

df["Performance_Category"] = pd.cut(

    df["Final_Percentage"],

    bins=[
        0,
        40,
        60,
        75,
        90,
        100
    ],

    labels=[
        "Fail",
        "Average",
        "Good",
        "Very Good",
        "Excellent"
    ],

    include_lowest=True

)


# ============================================================
# AT-RISK
# ============================================================

df["At_Risk"] = np.where(

    (
        (df["Final_Percentage"] < 50)
        |
        (df["Failures"] >= 2)
        |
        (df["Absences"] >= 30)
    ),

    "Yes",

    "No"

)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Student Performance Analyzer"
)

st.sidebar.write(
    "Academic Analytics & Machine Learning Dashboard"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Learner Profiles",
        "Model Performance",
        "Prediction"
    ]

)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Total Students: {len(df):,}"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Student Performance Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Overview of student academic performance, '
        'attendance, study behavior and risk indicators.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # KPIs
    # ========================================================

    pass_rate = (
        (df["Pass_Fail"] == "Pass").mean()
        * 100
    )

    average_final = (
        df["Final_Percentage"].mean()
    )

    at_risk_students = (
        df["At_Risk"] == "Yes"
    ).sum()

    average_absences = (
        df["Absences"].mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Pass Rate</div>
                <div class="kpi-value">
                    {pass_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Average Final Score
                </div>
                <div class="kpi-value">
                    {average_final:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    At-Risk Students
                </div>
                <div class="kpi-value">
                    {at_risk_students:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Average Absences
                </div>
                <div class="kpi-value">
                    {average_absences:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # PERFORMANCE CATEGORY
    # ========================================================

    st.subheader(
        "Performance Category Distribution"
    )


    category_order = [

        "Fail",
        "Average",
        "Good",
        "Very Good",
        "Excellent"

    ]


    category_counts = (

        df["Performance_Category"]

        .value_counts()

        .reindex(
            category_order,
            fill_value=0
        )

        .reset_index()

    )


    category_counts.columns = [

        "Performance Category",
        "Number of Students"

    ]


    fig = px.bar(

        category_counts,

        x="Performance Category",

        y="Number of Students",

        category_orders={

            "Performance Category":
            category_order

        },

        text="Number of Students",

        title="Students by Performance Category"

    )


    fig.update_traces(
        textposition="outside"
    )


    fig = style_chart(
        fig,
        450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.caption(
        "0–40 = Fail | "
        "40–60 = Average | "
        "60–75 = Good | "
        "75–90 = Very Good | "
        "90–100 = Excellent"
    )


    # ========================================================
    # FINAL PERCENTAGE + SUBJECT SCORES
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Final Percentage Distribution"
        )


        fig = px.histogram(

            df,

            x="Final_Percentage",

            nbins=20,

            title="Distribution of Final Percentage"

        )


        fig = style_chart(
            fig,
            400
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Average Subject Scores"
        )


        subject_means = pd.DataFrame({

            "Subject": [

                "Mathematics",
                "Science",
                "English"

            ],

            "Average Score": [

                df["Math_Score"].mean(),

                df["Science_Score"].mean(),

                df["English_Score"].mean()

            ]

        })


        fig = px.bar(

            subject_means,

            x="Subject",

            y="Average Score",

            text="Average Score",

            title="Average Score by Subject"

        )


        fig.update_traces(

            texttemplate="%{text:.2f}",

            textposition="outside"

        )


        fig = style_chart(
            fig,
            400
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # STUDY TIME
    # ========================================================

    st.subheader(
        "Average Final Percentage by Study Time"
    )


    study_time_avg = (

        df.groupby(
            "Study Time"
        )["Final_Percentage"]

        .mean()

        .reset_index()

    )


    fig = px.bar(

        study_time_avg,

        x="Study Time",

        y="Final_Percentage",

        text="Final_Percentage",

        title="Average Final Percentage by Study Time"

    )


    fig.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside"

    )


    fig = style_chart(
        fig,
        450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # ATTENDANCE
    # ========================================================

    st.subheader(
        "Average Final Percentage by Attendance"
    )


    df["Attendance_Group"] = pd.cut(

        df["Attendance_Percentage"],

        bins=[
            0,
            60,
            75,
            90,
            100
        ],

        labels=[
            "0–60%",
            "60–75%",
            "75–90%",
            "90–100%"
        ],

        include_lowest=True

    )


    attendance_avg = (

        df.groupby(
            "Attendance_Group",
            observed=False
        )["Final_Percentage"]

        .mean()

        .reset_index()

    )


    fig = px.bar(

        attendance_avg,

        x="Attendance_Group",

        y="Final_Percentage",

        text="Final_Percentage",

        title="Average Final Percentage by Attendance Group"

    )


    fig.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside"

    )


    fig = style_chart(
        fig,
        450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # FAILURE + PREVIOUS SCORE
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Average Final Percentage by Failures"
        )


        failure_avg = (

            df.groupby(
                "Failures"
            )["Final_Percentage"]

            .mean()

            .reset_index()

        )


        fig = px.bar(

            failure_avg,

            x="Failures",

            y="Final_Percentage",

            text="Final_Percentage",

            title="Final Percentage by Number of Failures"

        )


        fig.update_traces(

            texttemplate="%{text:.2f}",

            textposition="outside"

        )


        fig = style_chart(
            fig,
            400
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Previous Year Score Analysis"
        )


        df["Previous_Score_Group"] = pd.cut(

            df["Previous_Year_Score"],

            bins=[
                0,
                40,
                60,
                75,
                90,
                100
            ],

            labels=[
                "0–40",
                "40–60",
                "60–75",
                "75–90",
                "90–100"
            ],

            include_lowest=True

        )


        previous_score_avg = (

            df.groupby(
                "Previous_Score_Group",
                observed=False
            )["Final_Percentage"]

            .mean()

            .reset_index()

        )


        fig = px.bar(

            previous_score_avg,

            x="Previous_Score_Group",

            y="Final_Percentage",

            text="Final_Percentage",

            title="Final Percentage by Previous Year Score"

        )


        fig.update_traces(

            texttemplate="%{text:.2f}",

            textposition="outside"

        )


        fig = style_chart(
            fig,
            400
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # STUDY TIME GROUP
    # ========================================================

    st.subheader(
        "Study Time Group Analysis"
    )


    df["Study_Time_Group"] = pd.cut(

        df["Study Time"],

        bins=[
            0,
            2,
            4,
            6
        ],

        labels=[
            "0.5–2 hours",
            "2–4 hours",
            "4–6 hours"
        ],

        include_lowest=True

    )


    study_group = (

        df.groupby(
            "Study_Time_Group",
            observed=False
        )["Final_Percentage"]

        .mean()

        .reset_index()

    )


    fig = px.bar(

        study_group,

        x="Study_Time_Group",

        y="Final_Percentage",

        text="Final_Percentage",

        title="Average Final Percentage by Study Time Group"

    )


    fig.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside"

    )


    fig = style_chart(
        fig,
        400
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # INTERNET ACCESS
    # ========================================================

    st.subheader(
        "Internet Access Analysis"
    )


    internet_avg = (

        df.groupby(
            "Internet_Access"
        )["Final_Percentage"]

        .mean()

        .reset_index()

    )


    fig = px.bar(

        internet_avg,

        x="Internet_Access",

        y="Final_Percentage",

        text="Final_Percentage",

        title="Average Final Percentage by Internet Access"

    )


    fig.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside"

    )


    fig = style_chart(
        fig,
        400
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # AT-RISK
    # ========================================================

    st.subheader(
        "At-Risk Student Distribution"
    )


    risk_counts = (

        df["At_Risk"]

        .value_counts()

        .reset_index()

    )


    risk_counts.columns = [

        "Risk Status",

        "Number of Students"

    ]


    fig = px.pie(

        risk_counts,

        names="Risk Status",

        values="Number of Students",

        title="At-Risk vs Not At-Risk Students"

    )


    fig.update_layout(

        height=450,

        paper_bgcolor="#ffffff",

        font=dict(
            color="#0f172a"
        )

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# LEARNER PROFILES
# ============================================================

elif page == "Learner Profiles":

    st.markdown(
        '<div class="main-title">'
        'Learner Profiles'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Descriptive student profiles generated using K-Means.'
        '</div>',
        unsafe_allow_html=True
    )


    cluster_features = [

        "Study Time",
        "Absences",
        "Previous_Year_Score"

    ]


    cluster_data = (

        df[
            cluster_features
        ]

        .dropna()

        .copy()

    )


    scaler = StandardScaler()


    scaled_features = scaler.fit_transform(
        cluster_data
    )


    # ========================================================
    # SILHOUETTE
    # ========================================================

    silhouette_results = []


    for k in [3, 4, 5]:

        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )


        labels = model.fit_predict(
            scaled_features
        )


        score = silhouette_score(

            scaled_features,

            labels

        )


        silhouette_results.append({

            "K": k,

            "Silhouette Score": score

        })


    silhouette_df = pd.DataFrame(
        silhouette_results
    )


    st.subheader(
        "K-Means Cluster Selection"
    )


    fig = px.bar(

        silhouette_df,

        x="K",

        y="Silhouette Score",

        text="Silhouette Score",

        title="Silhouette Score for Different K Values"

    )


    fig.update_traces(

        texttemplate="%{text:.3f}",

        textposition="outside"

    )


    fig = style_chart(
        fig,
        400
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # K = 5
    # ========================================================

    kmeans = KMeans(

        n_clusters=5,

        random_state=42,

        n_init=10

    )


    cluster_labels = kmeans.fit_predict(
        scaled_features
    )


    cluster_data["Cluster"] = cluster_labels


    df.loc[
        cluster_data.index,
        "Cluster"
    ] = cluster_labels


    # ========================================================
    # CLUSTER PROFILE
    # ========================================================

    st.subheader(
        "Cluster Profiles"
    )


    cluster_profile = (

        df.groupby(
            "Cluster",
            observed=False
        )

        .agg(

            Students=(
                "Student_ID",
                "count"
            ),

            Study_Time=(
                "Study Time",
                "mean"
            ),

            Absences=(
                "Absences",
                "mean"
            ),

            Previous_Score=(
                "Previous_Year_Score",
                "mean"
            ),

            Final_Percentage=(
                "Final_Percentage",
                "mean"
            )

        )

        .reset_index()

    )


    cluster_profile = (
        cluster_profile.round(2)
    )


    st.dataframe(

        cluster_profile,

        use_container_width=True,

        hide_index=True

    )


    st.info(

        "Clusters are descriptive profiles. "
        "They should not be interpreted as objectively "
        "good or bad student groups."

    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation of regression and classification models.'
        '</div>',
        unsafe_allow_html=True
    )


    features = [

        "Age",
        "Gender",
        "Class",
        "Study Time",
        "Absences",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities",
        "Previous_Year_Score"

    ]


    numeric_features = [

        "Age",
        "Class",
        "Study Time",
        "Absences",
        "Previous_Year_Score"

    ]


    categorical_features = [

        "Gender",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities"

    ]


    X = df[features]

    y_reg = df[
        "Final_Percentage"
    ]

    y_cls = df[
        "Pass_Fail"
    ]


    # ========================================================
    # PREPROCESSOR
    # ========================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features

            ),

            (
                "numerical",

                "passthrough",

                numeric_features

            )

        ]

    )


    # ========================================================
    # REGRESSION
    # ========================================================

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = (

        train_test_split(

            X,

            y_reg,

            test_size=0.20,

            random_state=42

        )

    )


    ridge_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            Ridge(alpha=1.0)
        )

    ])


    rf_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestRegressor(

                n_estimators=200,

                random_state=42

            )

        )

    ])


    xgb_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            XGBRegressor(

                n_estimators=200,

                learning_rate=0.05,

                max_depth=4,

                random_state=42,

                objective="reg:squarederror"

            )

        )

    ])


    ridge_model.fit(
        X_train_reg,
        y_train_reg
    )

    rf_model.fit(
        X_train_reg,
        y_train_reg
    )

    xgb_model.fit(
        X_train_reg,
        y_train_reg
    )


    ridge_pred = ridge_model.predict(
        X_test_reg
    )

    rf_pred = rf_model.predict(
        X_test_reg
    )

    xgb_pred = xgb_model.predict(
        X_test_reg
    )


    regression_results = pd.DataFrame({

        "Model": [

            "Ridge Regression",

            "Random Forest Regressor",

            "XGBoost Regressor"

        ],

        "MAE": [

            mean_absolute_error(
                y_test_reg,
                ridge_pred
            ),

            mean_absolute_error(
                y_test_reg,
                rf_pred
            ),

            mean_absolute_error(
                y_test_reg,
                xgb_pred
            )

        ],

        "RMSE": [

            np.sqrt(
                mean_squared_error(
                    y_test_reg,
                    ridge_pred
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test_reg,
                    rf_pred
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test_reg,
                    xgb_pred
                )
            )

        ],

        "R2": [

            r2_score(
                y_test_reg,
                ridge_pred
            ),

            r2_score(
                y_test_reg,
                rf_pred
            ),

            r2_score(
                y_test_reg,
                xgb_pred
            )

        ]

    })


    st.subheader(
        "Regression Models"
    )


    st.dataframe(

        regression_results.round(4),

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = (

        train_test_split(

            X,

            y_cls,

            test_size=0.20,

            random_state=42,

            stratify=y_cls

        )

    )


    logistic_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            LogisticRegression(
                max_iter=1000
            )

        )

    ])


    rf_classifier = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestClassifier(

                n_estimators=200,

                random_state=42,

                class_weight="balanced"

            )

        )

    ])


    logistic_model.fit(
        X_train_cls,
        y_train_cls
    )

    rf_classifier.fit(
        X_train_cls,
        y_train_cls
    )


    logistic_pred = logistic_model.predict(
        X_test_cls
    )

    rf_cls_pred = rf_classifier.predict(
        X_test_cls
    )


    classification_results = pd.DataFrame({

        "Model": [

            "Logistic Regression",

            "Random Forest Classifier"

        ],

        "Accuracy": [

            accuracy_score(
                y_test_cls,
                logistic_pred
            ),

            accuracy_score(
                y_test_cls,
                rf_cls_pred
            )

        ],

        "Fail Precision": [

            precision_score(
                y_test_cls,
                logistic_pred,
                pos_label="Fail",
                zero_division=0
            ),

            precision_score(
                y_test_cls,
                rf_cls_pred,
                pos_label="Fail",
                zero_division=0
            )

        ],

        "Fail Recall": [

            recall_score(
                y_test_cls,
                logistic_pred,
                pos_label="Fail",
                zero_division=0
            ),

            recall_score(
                y_test_cls,
                rf_cls_pred,
                pos_label="Fail",
                zero_division=0
            )

        ],

        "Fail F1": [

            f1_score(
                y_test_cls,
                logistic_pred,
                pos_label="Fail",
                zero_division=0
            ),

            f1_score(
                y_test_cls,
                rf_cls_pred,
                pos_label="Fail",
                zero_division=0
            )

        ]

    })


    st.subheader(
        "Classification Models"
    )


    st.dataframe(

        classification_results.round(4),

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "Random Forest Confusion Matrix"
    )


    cm = confusion_matrix(

        y_test_cls,

        rf_cls_pred,

        labels=[
            "Fail",
            "Pass"
        ]

    )


    cm_df = pd.DataFrame(

        cm,

        index=[
            "Actual Fail",
            "Actual Pass"
        ],

        columns=[
            "Predicted Fail",
            "Predicted Pass"
        ]

    )


    st.dataframe(

        cm_df,

        use_container_width=True

    )


    st.warning(

        "The Pass class is dominant. "
        "Therefore, accuracy alone should not be used "
        "to judge classification performance."

    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.markdown(
        '<div class="main-title">'
        'Student Performance Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter student information to generate a model-based estimate.'
        '</div>',
        unsafe_allow_html=True
    )


    features = [

        "Age",
        "Gender",
        "Class",
        "Study Time",
        "Absences",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities",
        "Previous_Year_Score"

    ]


    numeric_features = [

        "Age",
        "Class",
        "Study Time",
        "Absences",
        "Previous_Year_Score"

    ]


    categorical_features = [

        "Gender",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities"

    ]


    X = df[features]

    y_reg = df[
        "Final_Percentage"
    ]

    y_cls = df[
        "Pass_Fail"
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features

            ),

            (
                "numerical",

                "passthrough",

                numeric_features

            )

        ]

    )


    # ========================================================
    # PREDICTION MODELS
    # ========================================================

    ridge_prediction_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            Ridge(alpha=1.0)
        )

    ])


    logistic_prediction_model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            LogisticRegression(
                max_iter=1000
            )

        )

    ])


    ridge_prediction_model.fit(
        X,
        y_reg
    )

    logistic_prediction_model.fit(
        X,
        y_cls
    )


    # ========================================================
    # INPUTS
    # ========================================================

    st.subheader(
        "Student Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(

            "Age",

            min_value=10,

            max_value=30,

            value=18

        )


        student_class = st.number_input(

            "Class",

            min_value=1,

            max_value=12,

            value=10

        )


        study_time = st.number_input(

            "Study Time (Hours/Day)",

            min_value=0.0,

            max_value=12.0,

            value=3.0,

            step=0.5

        )


    with col2:

        absences = st.number_input(

            "Absences",

            min_value=0.0,

            max_value=100.0,

            value=20.0,

            step=1.0

        )


        previous_score = st.number_input(

            "Previous Year Score",

            min_value=0.0,

            max_value=100.0,

            value=60.0,

            step=1.0

        )


        gender = st.selectbox(

            "Gender",

            sorted(
                df["Gender"]
                .dropna()
                .unique()
                .tolist()
            )

        )


    with col3:

        parental_education = st.selectbox(

            "Parental Education",

            sorted(
                df["Parental_Education"]
                .dropna()
                .unique()
                .tolist()
            )

        )


        internet_access = st.selectbox(

            "Internet Access",

            sorted(
                df["Internet_Access"]
                .dropna()
                .unique()
                .tolist()
            )

        )


        extracurricular = st.selectbox(

            "Extracurricular Activities",

            sorted(
                df["Extracurricular_Activities"]
                .dropna()
                .unique()
                .tolist()
            )

        )


    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(

        "Predict Student Performance",

        use_container_width=True

    ):

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "Class": [student_class],

            "Study Time": [study_time],

            "Absences": [absences],

            "Parental_Education": [
                parental_education
            ],

            "Internet_Access": [
                internet_access
            ],

            "Extracurricular_Activities": [
                extracurricular
            ],

            "Previous_Year_Score": [
                previous_score
            ]

        })


        # ----------------------------------------------------
        # REGRESSION
        # ----------------------------------------------------

        predicted_percentage = (

            ridge_prediction_model

            .predict(
                input_data
            )[0]

        )


        predicted_percentage = np.clip(

            predicted_percentage,

            0,

            100

        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        predicted_result = (

            logistic_prediction_model

            .predict(
                input_data
            )[0]

        )


        probabilities = (

            logistic_prediction_model

            .predict_proba(
                input_data
            )[0]

        )


        classes = (

            logistic_prediction_model

            .classes_

        )


        probability_dict = dict(

            zip(
                classes,
                probabilities
            )

        )


        pass_probability = (

            probability_dict.get(
                "Pass",
                0
            )
            * 100

        )


        fail_probability = (

            probability_dict.get(
                "Fail",
                0
            )
            * 100

        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )


        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(

                "Predicted Final Percentage",

                f"{predicted_percentage:.2f}%"

            )


        with col2:

            st.metric(

                "Predicted Result",

                str(predicted_result)

            )


        with col3:

            st.metric(

                "Pass Probability",

                f"{pass_probability:.2f}%"

            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.subheader(
            "Classification Probabilities"
        )


        probability_df = pd.DataFrame({

            "Result": [
                "Pass",
                "Fail"
            ],

            "Probability": [

                pass_probability,

                fail_probability

            ]

        })


        fig = px.bar(

            probability_df,

            x="Result",

            y="Probability",

            text="Probability",

            title="Pass/Fail Probability"

        )


        fig.update_traces(

            texttemplate="%{text:.2f}%",

            textposition="outside"

        )


        fig.update_yaxes(
            range=[0, 100]
        )


        fig = style_chart(
            fig,
            400
        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


        st.info(

            "These predictions are model outputs from "
            "the academic prototype. They should not be "
            "treated as official academic decisions or "
            "validated individual-level risk probabilities."

        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student Performance Analyzer | "
    "Python | Pandas | Scikit-learn | XGBoost | Streamlit"
)