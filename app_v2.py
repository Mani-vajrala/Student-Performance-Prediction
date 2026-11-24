from pathlib import Path
import math
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI | Version 2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS AND FEATURE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "StudentPerformanceFactors.csv"

FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

CATEGORY_OPTIONS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Female", "Male"],
}


# ============================================================
# 3. CUSTOM STYLING
# ============================================================

import altair as alt

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --page-bg: #F4F7FC;
        --card-bg: #FFFFFF;
        --sidebar-bg: #102B55;
        --ink: #142B4A;
        --ink-muted: #4A5D7A;
        --on-dark: #FFFFFF;
        --on-dark-muted: #C9D8F2;
        --line: #DCE5F1;
        --brand: #173B75;
        --brand-hover: #24549A;
        --hover-bg: #E6EEFB;
    }

    /* ---------- Page and header ---------- */
    .stApp,
    [data-testid="stMain"] {
        background-color: var(--page-bg);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background-color: var(--page-bg);
    }

    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] * {
        color: var(--ink) !important;
    }

    /* ---------- Main text: dark on light ---------- */
    [data-testid="stMain"] :is(
        h1, h2, h3, h4, h5, h6, p, li, span, label, td, th, strong, em,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetricLabel"],
        [data-testid="stWidgetLabel"]
    ) {
        color: var(--ink);
    }

    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
        color: var(--ink-muted) !important;
    }

    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* ---------- Sidebar: white on dark blue ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
    }

    [data-testid="stSidebar"] * {
        color: var(--on-dark) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: var(--on-dark-muted) !important;
    }

    /* ---------- Cards ---------- */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(16, 43, 85, 0.06);
    }

    div[data-testid="stMetric"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: var(--ink) !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700;
    }

    div[data-testid="stForm"] {
        background-color: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(16, 43, 85, 0.06);
    }

    /* ---------- Main buttons: white text on dark blue ---------- */
    .stButton > button,
    .stFormSubmitButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-secondaryFormSubmit"],
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: var(--brand) !important;
        border: 1px solid var(--brand) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button *,
    .stFormSubmitButton > button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-secondaryFormSubmit"] *,
    [data-testid="stBaseButton-primaryFormSubmit"] * {
        color: var(--on-dark) !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    [data-testid="stBaseButton-secondaryFormSubmit"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background-color: var(--brand-hover) !important;
        border-color: var(--brand-hover) !important;
    }

    /* ---------- Input fields: dark text on white ---------- */
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-baseweb="select"] > div {
        background-color: var(--card-bg) !important;
        border-color: var(--line) !important;
    }

    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    [data-baseweb="select"] *,
    [data-testid="stNumberInput"] input {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        background-color: transparent !important;
    }

    [data-baseweb="select"] svg {
        fill: var(--ink) !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: var(--hover-bg) !important;
        border-color: var(--line) !important;
    }

    [data-testid="stNumberInput"] button * {
        color: var(--ink) !important;
        fill: var(--ink) !important;
    }

    /* ---------- Popups: dropdown lists, top-right menu, tooltips ---------- */
    [data-baseweb="popover"],
    [data-baseweb="popover"] :is(div, ul, ol, li, span, p, a, label, button) {
        background-color: var(--card-bg) !important;
        color: var(--ink) !important;
    }

    [data-baseweb="popover"] svg {
        fill: var(--ink) !important;
        color: var(--ink) !important;
    }

    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li:hover *,
    [data-baseweb="popover"] li[aria-selected="true"],
    [data-baseweb="popover"] li[aria-selected="true"] * {
        background-color: var(--hover-bg) !important;
        color: var(--ink) !important;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div {
        background-color: #E8F1FF !important;
        border-radius: 8px;
    }

    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) > div {
        background-color: #FFF4D6 !important;
    }

    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) > div {
        background-color: #FDE8E8 !important;
    }

    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) > div {
        background-color: #E3F6E8 !important;
    }

    [data-testid="stAlert"] * {
        color: var(--ink) !important;
    }

    /* ---------- Code blocks and error traces ---------- */
    [data-testid="stCode"],
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {
        background-color: #EEF3FA !important;
    }

    [data-testid="stCode"] pre *,
    [data-testid="stCode"] code * {
        color: var(--ink) !important;
    }

    [data-testid="stException"],
    [data-testid="stException"] * {
        background-color: #FDE8E8 !important;
        color: var(--ink) !important;
    }

        /* ---------- Number and text input boxes: force white box + dark text ---------- */
    [data-testid="stNumberInputContainer"],
    [data-testid="stNumberInputContainer"] > div,
    [data-testid="stNumberInputContainer"] [data-baseweb],
    [data-testid="stNumberInputContainer"] [data-baseweb] > div,
    [data-testid="stNumberInputField"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextInputRootElement"] > div,
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea {
        background-color: #FFFFFF !important;
        color: #142B4A !important;
        -webkit-text-fill-color: #142B4A !important;
        caret-color: #142B4A !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInputContainer"] {
        border: 1px solid #DCE5F1 !important;
        border-radius: 8px !important;
    }

    /* the small − and + buttons next to the number */
    [data-testid="stNumberInputContainer"] button {
        background-color: #E6EEFB !important;
        border-color: #DCE5F1 !important;
    }

    [data-testid="stNumberInputContainer"] button,
    [data-testid="stNumberInputContainer"] button * {
        color: #142B4A !important;
        fill: #142B4A !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart):
    """Give charts fixed colours so they never depend on the Streamlit theme."""
    return (
        chart.configure(background="#FFFFFF")
        .configure_axis(
            labelColor="#142B4A",
            titleColor="#142B4A",
            gridColor="#DCE5F1",
            domainColor="#9FB2CF",
            tickColor="#9FB2CF",
        )
        .configure_view(strokeWidth=0)
    )

# ============================================================
# 4. LOAD MODEL AND DATASET
# ============================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_dataset(data_path):
    return pd.read_csv(data_path)


model = None
df = None
model_error = None
data_error = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as error:
        model_error = str(error)
else:
    model_error = f"Model file not found: {MODEL_PATH}"

if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as error:
        data_error = str(error)
else:
    data_error = f"Dataset file not found: {DATA_PATH}"


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.title("🎓 Student Performance AI")
    st.caption("Machine Learning Prediction System")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        ["Dashboard", "Student Prediction", "About Project"],
        index=0,
    )

    st.divider()
    st.caption("VERSION 2.0")
    st.caption("Academic Machine Learning Project")


# ============================================================
# 6. SHARED VALIDATION MESSAGES
# ============================================================

if model_error:
    st.error("The trained model could not be loaded.")
    st.code(model_error)

if data_error:
    st.error("The dataset could not be loaded.")
    st.code(data_error)


# ============================================================
# 7. DASHBOARD PAGE
# ============================================================

if selected_page == "Dashboard":
    st.title("Student Performance Dashboard")
    st.caption(
        "Explore the Student Performance Factors dataset and the "
        "machine learning prediction system."
    )

    if df is None:
        st.warning(
            "The dashboard needs StudentPerformanceFactors.csv. "
            "Place it inside the project's dataset folder."
        )
    elif "Exam_Score" not in df.columns:
        st.error("The dataset does not contain the required Exam_Score column.")
    else:
        total_records = len(df)
        average_score = float(df["Exam_Score"].mean())
        highest_score = float(df["Exam_Score"].max())
        lowest_score = float(df["Exam_Score"].min())

        metric_columns = st.columns(4)
        metric_columns[0].metric("Total Students", f"{total_records:,}")
        metric_columns[1].metric("Average Exam Score", f"{average_score:.2f}")
        metric_columns[2].metric("Highest Exam Score", f"{highest_score:.2f}")
        metric_columns[3].metric("Lowest Exam Score", f"{lowest_score:.2f}")

        st.divider()
        st.subheader("Dataset Overview")

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("**Exam Score Distribution**")
            score_bins = pd.cut(
                df["Exam_Score"],
                bins=8,
                include_lowest=True,
            )
            distribution = (
                score_bins.value_counts()
                .sort_index()
                .rename_axis("Score Range")
                .to_frame("Number of Students")
            )
            distribution.index = distribution.index.astype(str)
            bar_chart = (
                alt.Chart(distribution.reset_index())
                .mark_bar(color="#24549A", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Score Range:N", sort=None, title="Score Range",
                            axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("Number of Students:Q", title="Number of Students"),
                    tooltip=["Score Range", "Number of Students"],
                )
                .properties(height=250)
            )
            st.altair_chart(style_chart(bar_chart), theme=None)

        with chart_right:
            st.markdown("**Average Score by Study Hours**")
            if "Hours_Studied" in df.columns:
                study_summary = (
                    df.groupby("Hours_Studied", as_index=True)["Exam_Score"]
                    .mean()
                    .sort_index()
                    .to_frame("Average Exam Score")
                )
                line_chart = (
                    alt.Chart(study_summary.reset_index())
                    .mark_line(color="#24549A", strokeWidth=3, point=True)
                    .encode(
                        x=alt.X("Hours_Studied:Q", title="Hours Studied"),
                        y=alt.Y("Average Exam Score:Q", title="Average Exam Score",
                                scale=alt.Scale(zero=False)),
                        tooltip=["Hours_Studied", "Average Exam Score"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(style_chart(line_chart), theme=None)
            else:
                st.info("Hours_Studied is not available in this dataset.")

        st.divider()
        st.subheader("Dataset Information")

        info_left, info_right = st.columns(2)
        with info_left:
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Input features used:** {len(FEATURE_COLUMNS)}")
            st.write("**Target variable:** Exam_Score")

        with info_right:
            missing_values = int(df.isnull().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            st.write(f"**Missing values:** {missing_values}")
            st.write(f"**Duplicate rows:** {duplicate_rows}")
            st.write("**Model:** Linear Regression")
            st.write("**Train-test split:** 70:30")

        st.info(
            "Use the Student Prediction page to estimate an individual "
            "student's examination score."
        )


# ============================================================
# 8. STUDENT PREDICTION PAGE
# ============================================================

elif selected_page == "Student Prediction":
    st.title("Student Exam Score Prediction")
    st.caption(
        "Enter the student's academic, lifestyle, family and school details "
        "to estimate the examination score."
    )

    if model is None:
        st.warning(
            "Prediction is unavailable because the trained model could not "
            "be loaded. Check the models folder and model file."
        )
    else:
        def category_input(label, column_name):
            options = CATEGORY_OPTIONS[column_name]
            return st.selectbox(label, options, index=1 if len(options) == 3 else 0)

        with st.form("student_prediction_form"):
            st.subheader("Academic Information")
            academic_left, academic_right = st.columns(2)

            with academic_left:
                hours_studied = st.number_input(
                    "Hours Studied per Week",
                    min_value=1.0,
                    max_value=44.0,
                    value=20.0,
                    step=1.0,
                )
                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=60.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                )
                previous_scores = st.number_input(
                    "Previous Scores",
                    min_value=50.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                )
                tutoring_sessions = st.number_input(
                    "Tutoring Sessions",
                    min_value=0,
                    max_value=8,
                    value=1,
                    step=1,
                )

            with academic_right:
                access_resources = category_input(
                    "Access to Resources", "Access_to_Resources"
                )
                extracurricular = category_input(
                    "Extracurricular Activities", "Extracurricular_Activities"
                )
                motivation = category_input(
                    "Motivation Level", "Motivation_Level"
                )
                teacher_quality = category_input(
                    "Teacher Quality", "Teacher_Quality"
                )

            st.divider()
            st.subheader("Lifestyle and Personal Factors")
            lifestyle_left, lifestyle_right = st.columns(2)

            with lifestyle_left:
                sleep_hours = st.number_input(
                    "Sleep Hours per Day",
                    min_value=4.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.5,
                )
                physical_activity = st.number_input(
                    "Physical Activity (hours per week)",
                    min_value=0,
                    max_value=6,
                    value=3,
                    step=1,
                )
                internet_access = category_input(
                    "Internet Access", "Internet_Access"
                )

            with lifestyle_right:
                learning_disabilities = category_input(
                    "Learning Disabilities", "Learning_Disabilities"
                )
                gender = category_input("Gender", "Gender")

            st.divider()
            st.subheader("Family and School Environment")
            family_left, family_right = st.columns(2)

            with family_left:
                parental_involvement = category_input(
                    "Parental Involvement", "Parental_Involvement"
                )
                parental_education = category_input(
                    "Parental Education", "Parental_Education_Level"
                )
                family_income = category_input(
                    "Family Income", "Family_Income"
                )
                distance = category_input(
                    "Distance from Home", "Distance_from_Home"
                )

            with family_right:
                school_type = category_input("School Type", "School_Type")
                peer_influence = category_input(
                    "Peer Influence", "Peer_Influence"
                )

            submitted = st.form_submit_button(
                "Generate Exam Score Prediction",
                use_container_width=True,
            )

        if submitted:
            input_values = {
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Parental_Involvement": parental_involvement,
                "Access_to_Resources": access_resources,
                "Extracurricular_Activities": extracurricular,
                "Sleep_Hours": sleep_hours,
                "Previous_Scores": previous_scores,
                "Motivation_Level": motivation,
                "Internet_Access": internet_access,
                "Tutoring_Sessions": tutoring_sessions,
                "Family_Income": family_income,
                "Teacher_Quality": teacher_quality,
                "School_Type": school_type,
                "Peer_Influence": peer_influence,
                "Physical_Activity": physical_activity,
                "Learning_Disabilities": learning_disabilities,
                "Parental_Education_Level": parental_education,
                "Distance_from_Home": distance,
                "Gender": gender,
            }

            try:
                input_data = pd.DataFrame(
                    [input_values],
                    columns=FEATURE_COLUMNS,
                )

                prediction = float(model.predict(input_data)[0])

                if not math.isfinite(prediction):
                    raise ValueError(
                        "The model returned a non-finite prediction."
                    )
                
                prediction = max(0.0, min(100.0, prediction))

                st.divider()
                with st.container(border=True):
                    st.markdown("**PREDICTED EXAM SCORE**")
                    st.markdown(f"# {prediction:.2f} / 100")
                    st.caption(
                        "Estimated using the trained machine learning model."
                    )

            except Exception as error:
                st.error(
                    "Unable to generate the prediction. "
                    "Check the model file and its scikit-learn compatibility."
                )
                st.exception(error)


# ============================================================
# 9. ABOUT PROJECT PAGE
# ============================================================

elif selected_page == "About Project":
    st.title("About the Project")
    st.caption(
        "The dataset, machine learning workflow and final model evaluation."
    )

    st.subheader("Project Information")
    st.write(
        "The Student Performance Prediction System estimates student "
        "examination scores using academic, personal, family and "
        "school-related factors."
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        with st.container(border=True):
            st.markdown("**Dataset**")
            st.write("- Dataset: Student Performance Factors")
            if df is not None:
                st.write(f"- Total records: {len(df):,}")
            else:
                st.write("- Total records: Dataset unavailable")
            st.write(f"- Input features: {len(FEATURE_COLUMNS)}")
            st.write("- Target variable: Exam_Score")
            st.write("- Train-test split: 70:30")

    with info_col2:
        with st.container(border=True):
            st.markdown("**Machine Learning**")
            st.write("- Selected model: Linear Regression")
            st.write("- Cross-validation: 5-Fold")
            st.write("- Hyperparameter tuning: GridSearchCV")
            st.write("- Preprocessing: Scikit-learn Pipeline")
            st.write("- Interface: Streamlit")

    st.divider()
    st.subheader("Final Model Evaluation")

    evaluation_columns = st.columns(3)
    evaluation_columns[0].metric("MAE", "0.4531")
    evaluation_columns[1].metric("RMSE", "1.7638")
    evaluation_columns[2].metric("R² Score", "0.7736")

    st.info(
        "The R² score indicates that approximately 77.36% of the variation "
        "in test-set examination scores is explained by the model. "
        "This is not classification accuracy."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()
st.caption(
    "STUDENT PERFORMANCE AI  |  VERSION 2.0  |  "
    "MACHINE LEARNING ACADEMIC PROJECT"
)
