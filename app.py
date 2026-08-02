import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="DataDoctor AI",
    page_icon="🩺",
    layout="wide"
)


# =====================================================
# SESSION STATE
# =====================================================

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = []

if "target" not in st.session_state:
    st.session_state.target = None

if "score" not in st.session_state:
    st.session_state.score = None


# =====================================================
# TITLE
# =====================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🩺 DataDoctor AI
    </h1>

    <p style="text-align:center;">
        AI-Powered Dataset Health & Machine Learning Assistant
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("🩺 DataDoctor AI")

    st.write(
        "Analyze, clean and prepare datasets "
        "for machine learning."
    )

    st.divider()

    st.write("### 🔍 Features")

    st.write("✅ Dataset Analysis")
    st.write("✅ Missing Value Detection")
    st.write("✅ Duplicate Detection")
    st.write("✅ Automatic Cleaning")
    st.write("✅ Random Forest ML")
    st.write("✅ Feature Importance")
    st.write("✅ Predictions")

    st.divider()

    st.caption(
        "Built with Python, Pandas, "
        "Scikit-learn & Streamlit"
    )


# =====================================================
# UPLOAD DATASET
# =====================================================

uploaded_file = st.file_uploader(
    "📁 Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(
        "✅ Dataset uploaded successfully!"
    )


    # =================================================
    # DATASET OVERVIEW
    # =================================================

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Duplicates",
            df.duplicated().sum()
        )

    with col4:
        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )


    # =================================================
    # DATA PREVIEW
    # =================================================

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )


    # =================================================
    # DATA HEALTH
    # =================================================

    st.subheader("🩺 Data Health")

    missing_values = df.isnull().sum().sum()

    duplicates = df.duplicated().sum()

    health_score = 100

    health_score -= missing_values * 2

    health_score -= duplicates * 5

    health_score = max(
        0,
        health_score
    )

    st.metric(
        "Data Health Score",
        f"{health_score}/100"
    )


    if health_score >= 80:

        st.success(
            "🟢 Dataset looks healthy."
        )

    elif health_score >= 60:

        st.warning(
            "🟡 Dataset needs some cleaning."
        )

    else:

        st.error(
            "🔴 Dataset needs major cleaning."
        )


    # =================================================
    # MISSING VALUES
    # =================================================

    st.subheader("⚠️ Missing Values")

    missing = df.isnull().sum()

    st.bar_chart(missing)


    # =================================================
    # STATISTICS
    # =================================================

    st.subheader("📈 Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


    # =================================================
    # AUTOMATIC CLEANING
    # =================================================

    st.subheader("🧹 Automatic Data Cleaning")

    if st.button(
        "🧹 Clean Dataset"
    ):

        cleaned_df = df.copy()

        missing_before = (
            cleaned_df.isnull().sum().sum()
        )

        duplicates_before = (
            cleaned_df.duplicated().sum()
        )


        # Remove duplicates

        cleaned_df = (
            cleaned_df.drop_duplicates()
        )


        # Fill numerical missing values

        numerical_columns = (
            cleaned_df
            .select_dtypes(include="number")
            .columns
        )


        for column in numerical_columns:

            if cleaned_df[column].isnull().any():

                median_value = (
                    cleaned_df[column].median()
                )

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(median_value)
                )


        # Fill text missing values

        text_columns = (
            cleaned_df
            .select_dtypes(exclude="number")
            .columns
        )


        for column in text_columns:

            if cleaned_df[column].isnull().any():

                mode_values = (
                    cleaned_df[column].mode()
                )

                if len(mode_values) > 0:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(mode_values[0])
                    )


        # Save cleaned dataset

        st.session_state.cleaned_df = (
            cleaned_df
        )


        # Reset old model

        st.session_state.model = None

        st.session_state.features = []

        st.session_state.target = None

        st.session_state.score = None


        missing_after = (
            cleaned_df.isnull().sum().sum()
        )

        duplicates_after = (
            cleaned_df.duplicated().sum()
        )


        st.success(
            "✅ Dataset cleaned successfully!"
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Missing Values Fixed",
                missing_before - missing_after
            )

        with col2:

            st.metric(
                "Duplicates Removed",
                duplicates_before - duplicates_after
            )


    # =================================================
    # SHOW CLEANED DATASET
    # =================================================

    if st.session_state.cleaned_df is not None:

        cleaned_df = (
            st.session_state.cleaned_df
        )


        st.subheader(
            "✨ Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )


        # =================================================
        # DOWNLOAD CLEAN DATA
        # =================================================

        csv_data = cleaned_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Clean Dataset",
            csv_data,
            "cleaned_dataset.csv",
            "text/csv"
        )


        # =================================================
        # MACHINE LEARNING
        # =================================================

        st.subheader(
            "🤖 Machine Learning"
        )


        numeric_columns = (
            cleaned_df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )


        if len(numeric_columns) >= 2:

            # ---------------------------------------------
            # TARGET
            # ---------------------------------------------

            target = st.selectbox(
                "🎯 Choose target to predict",
                numeric_columns
            )


            # ---------------------------------------------
            # FEATURES
            # ---------------------------------------------

            feature_options = [
                column
                for column in numeric_columns
                if column != target
            ]


            features = st.multiselect(
                "📌 Choose input features",
                feature_options,
                default=feature_options
            )


            if len(features) > 0:

                ml_data = cleaned_df[
                    features + [target]
                ].dropna()


                if len(ml_data) >= 8:

                    X = ml_data[features]

                    y = ml_data[target]


                    # -----------------------------------------
                    # TRAIN / TEST SPLIT
                    # -----------------------------------------

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )


                    # -----------------------------------------
                    # RANDOM FOREST
                    # -----------------------------------------

                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )


                    model.fit(
                        X_train,
                        y_train
                    )


                    # -----------------------------------------
                    # EVALUATION
                    # -----------------------------------------

                    predictions = model.predict(
                        X_test
                    )


                    score = r2_score(
                        y_test,
                        predictions
                    )


                    # Save model in session

                    st.session_state.model = model

                    st.session_state.features = features

                    st.session_state.target = target

                    st.session_state.score = score


                    # -----------------------------------------
                    # MODEL PERFORMANCE
                    # -----------------------------------------

                    st.subheader(
                        "🏆 Model Performance"
                    )


                    st.metric(
                        "R² Score",
                        f"{score:.2f}"
                    )


                    # -----------------------------------------
                    # FEATURE IMPORTANCE
                    # -----------------------------------------

                    st.subheader(
                        "🔍 Feature Importance"
                    )


                    importance = pd.DataFrame({

                        "Feature": features,

                        "Importance":
                            model.feature_importances_

                    })


                    importance = (
                        importance
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                    )


                    st.bar_chart(
                        importance.set_index(
                            "Feature"
                        )
                    )


                    # =================================================
                    # PREDICTION
                    # =================================================

                    st.subheader(
                        "🔮 Make a Prediction"
                    )


                    user_input = {}


                    for feature in features:

                        user_input[feature] = (
                            st.number_input(
                                f"Enter {feature}",
                                value=float(
                                    ml_data[feature].mean()
                                ),
                                key=f"input_{feature}"
                            )
                        )


                    if st.button(
                        "🔮 Predict",
                        key="predict_button"
                    ):

                        saved_model = (
                            st.session_state.model
                        )

                        saved_features = (
                            st.session_state.features
                        )

                        saved_target = (
                            st.session_state.target
                        )


                        input_data = pd.DataFrame(
                            [user_input]
                        )


                        input_data = input_data[
                            saved_features
                        ]


                        prediction = (
                            saved_model
                            .predict(input_data)
                        )


                        st.success(
                            f"🎯 Predicted "
                            f"{saved_target}: "
                            f"{prediction[0]:.2f}"
                        )


                else:

                    st.warning(
                        "⚠️ You need at least "
                        "8 clean rows for ML."
                    )


            else:

                st.info(
                    "Select at least one "
                    "input feature."
                )


        else:

            st.info(
                "You need at least two "
                "numerical columns for ML."
            )