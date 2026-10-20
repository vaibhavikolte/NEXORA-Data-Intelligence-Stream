
import hashlib
import io
import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="NEXORA | Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #102447 0%, transparent 38%),
        radial-gradient(ellipse at 95% 15%, #251344 0%, transparent 32%),
        #070b14;
    color: #eaf2ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1221, #111026);
    border-right: 1px solid #263858;
}

section[data-testid="stSidebar"] * {
    color: #dce9ff;
}

.nx-hero {
    position: relative;
    overflow: hidden;
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(125deg, #0b1428, #101e38 58%, #24143d);
    border: 1px solid #2c4e75;
    box-shadow: 0 0 30px rgba(0, 220, 255, .09),
                inset 0 0 30px rgba(89, 68, 255, .08);
    margin-bottom: 26px;
}

.nx-hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 229, 255, .055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, .055) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: linear-gradient(90deg, black, transparent 90%);
    pointer-events: none;
}

.nx-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
    margin-bottom: 25px;
}

.nx-logo {
    width: 54px;
    height: 54px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 25px;
    font-weight: 900;
    color: #06111b;
    background: linear-gradient(135deg, #55f7ff, #9a6bff);
    clip-path: polygon(25% 0, 75% 0, 100% 25%,
                      100% 75%, 75% 100%, 25% 100%,
                      0 75%, 0 25%);
    filter: drop-shadow(0 0 12px rgba(0, 229, 255, .45));
}

.nx-brand-name {
    color: #5cf6ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.nx-tag {
    display: inline-block;
    color: #b6a4ff;
    background: rgba(126, 87, 255, .10);
    border: 1px solid rgba(156, 124, 255, .5);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 14px;
    position: relative;
}

.nx-title {
    color: #f4f8ff !important;
    font-size: clamp(29px, 4vw, 46px) !important;
    line-height: 1.13 !important;
    letter-spacing: -1px;
    font-weight: 900 !important;
    margin: 0 0 14px !important;
    position: relative;
    text-shadow: 0 0 22px rgba(0, 229, 255, .12);
}

.nx-gradient {
    color: #5cf6ff;
    text-shadow: 0 0 18px rgba(0, 229, 255, .35);
}

.nx-desc {
    color: #a9bad5;
    font-size: 15px;
    line-height: 1.8;
    max-width: 720px;
    position: relative;
}

.nx-bottom {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
    position: relative;
}

.nx-chip {
    color: #b9faff;
    background: rgba(0, 229, 255, .055);
    border: 1px solid #28556b;
    padding: 8px 12px;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #101a2d, #0c1322);
    border: 1px solid #294361;
    border-radius: 14px;
    padding: 17px;
    box-shadow: 0 0 18px rgba(0, 229, 255, .045);
    transition: .2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #52eaf5;
    box-shadow: 0 0 20px rgba(0, 229, 255, .12);
}

div[data-testid="stMetricLabel"] {
    color: #9fb4d1;
}

div[data-testid="stMetricValue"] {
    color: #62f5ff;
    font-weight: 850;
}

.stMarkdown, p, label {
    color: #d6e2f5;
}

h1, h2, h3 {
    color: #f0f5ff;
}

hr {
    border-color: #263858;
}

.stButton > button,
.stDownloadButton > button {
    color: #eafaff;
    background: linear-gradient(110deg, #12364b, #28204b);
    border: 1px solid #3a8299;
    border-radius: 8px;
    font-weight: 750;
    transition: .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #5cf6ff;
    color: #5cf6ff;
    box-shadow: 0 0 15px rgba(0, 229, 255, .16);
}

div[data-testid="stFileUploader"] {
    background: rgba(15, 27, 47, .8);
    border: 1px dashed #3c7895;
    border-radius: 13px;
    padding: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263858;
    border-radius: 10px;
}

@media (prefers-reduced-motion: reduce) {
    div[data-testid="stMetric"],
    .stButton > button,
    .stDownloadButton > button {
        transition: none;
    }
}
</style>

<div class="nx-hero">
  <div class="nx-brand">
    <div class="nx-logo">N</div>
    <div class="nx-brand-name">NEXORA // DATA INTELLIGENCE</div>
  </div>

  <div class="nx-tag">◈ NEXT-GEN ANALYTICS WORKSPACE</div>

  <h1 class="nx-title">
    RAW DATA.<br>
    <span class="nx-gradient">REDEFINED.</span>
    INTELLIGENCE.
  </h1>

  <p class="nx-desc">
    Clean the noise. Validate the signal. Reveal the story hidden
    inside your data — from raw files to actionable insights.
  </p>

  <div class="nx-bottom">
    <span class="nx-chip">01 / INGEST</span>
    <span class="nx-chip">02 / CLEAN</span>
    <span class="nx-chip">03 / VALIDATE</span>
    <span class="nx-chip">04 / ANALYSE</span>
    <span class="nx-chip">05 / FORECAST</span>
    <span class="nx-chip">06 / EXPORT</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.success("NEXORA header test — if you can see this, Streamlit is running.")

# ============================================================
# 2. READ UPLOADED DATASET
# ============================================================
def read_uploaded_file(uploaded_file):
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    uploaded_file.seek(0)

    if extension == ".csv":
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")

    if extension in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV, XLSX or XLS file.")


# ============================================================
# 3. DETECT COLUMN TYPES
# ============================================================
def infer_column_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "Text"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"

    if pd.api.types.is_integer_dtype(series):
        return "Whole Number"

    if pd.api.types.is_numeric_dtype(series):
        return "Decimal Number"

    non_null = series.dropna()

    if non_null.empty:
        return "Text"

    name = str(series.name).lower()
    date_words = [
        "date", "time", "timestamp", "dob",
        "birth", "joining", "admission"
    ]

    if any(word in name for word in date_words):
        parsed_dates = pd.to_datetime(
            non_null, errors="coerce", dayfirst=True
        )
        if parsed_dates.notna().mean() >= 0.80:
            return "Date"

    numeric = pd.to_numeric(non_null, errors="coerce")

    if numeric.notna().mean() >= 0.95:
        valid = numeric.dropna()

        if len(valid) > 0 and np.isclose(
            valid % 1, 0
        ).all():
            return "Whole Number"

        return "Decimal Number"

    return "Text"


# ============================================================
# 4. CONVERT COLUMN TYPES AND CLEAN VALUES
# ============================================================
def convert_column(series, target_type, numeric_replacement=None):
    original = series.copy()
    invalid_count = 0

    if target_type == "Text":
        converted = original.map(
            lambda value: value.strip()
            if isinstance(value, str) else value
        ).astype("string")

        converted = converted.replace(
            r"^\s*$", pd.NA, regex=True
        )

    elif target_type == "Whole Number":
        numeric = pd.to_numeric(original, errors="coerce")

        invalid_mask = original.notna() & numeric.isna()
        fractional_mask = (
            numeric.notna() & ((numeric % 1).abs() > 1e-9)
        )

        invalid_count = int(
            invalid_mask.sum() + fractional_mask.sum()
        )

        converted = numeric.mask(
            fractional_mask, np.nan
        ).astype("Int64")

        if numeric_replacement is not None:
            converted = converted.fillna(
                int(numeric_replacement)
            )

    elif target_type == "Decimal Number":
        converted = pd.to_numeric(original, errors="coerce")

        invalid_count = int(
            (original.notna() & converted.isna()).sum()
        )

        if numeric_replacement is not None:
            converted = converted.fillna(
                float(numeric_replacement)
            )

    elif target_type == "Date":
        converted = pd.to_datetime(
            original, errors="coerce", dayfirst=True
        )

        invalid_count = int(
            (original.notna() & converted.isna()).sum()
        )

    else:
        converted = original.copy()

    return converted, invalid_count


# ============================================================
# 5. FIND COLUMNS FOR SALES ANALYTICS
# ============================================================
def find_matching_column(df, aliases):
    normalized = {}

    for col in df.columns:
        key = (
            str(col).strip().lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        normalized[" ".join(key.split())] = col

    for alias in aliases:
        key = (
            alias.strip().lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        key = " ".join(key.split())

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# 6. DATA QUALITY REPORT
# ============================================================
def build_quality_report(
    original_df,
    cleaned_df,
    invalid_counts,
    duplicates_removed=0,
):
    rows = []

    for col in original_df.columns:
        rows.append({
            "Column": col,
            "Original Type": str(original_df[col].dtype),
            "Final Type": str(cleaned_df[col].dtype),
            "Nulls Before": int(original_df[col].isna().sum()),
            "Nulls After": int(cleaned_df[col].isna().sum()),
            "Invalid Values Detected": int(
                invalid_counts.get(col, 0)
            ),
            "Unique Values Before": int(
                original_df[col].nunique(dropna=True)
            ),
            "Unique Values After": int(
                cleaned_df[col].nunique(dropna=True)
            ),
        })

    report = pd.DataFrame(rows)

    summary = {
        "Original Rows": int(len(original_df)),
        "Cleaned Rows": int(len(cleaned_df)),
        "Rows Removed": int(len(original_df) - len(cleaned_df)),
        "Original Columns": int(len(original_df.columns)),
        "Cleaned Columns": int(len(cleaned_df.columns)),
        "Null Cells Before": int(
            original_df.isna().sum().sum()
        ),
        "Null Cells After": int(
            cleaned_df.isna().sum().sum()
        ),
        "Duplicate Rows Found": int(
            original_df.duplicated().sum()
        ),
        "Duplicate Rows Removed": int(duplicates_removed),
        "Invalid Values Detected": int(
            sum(invalid_counts.values())
        ),
    }

    return report, summary


# ============================================================
# 7. EXCEL EXPORT
# ============================================================
def make_excel_file(
    cleaned_df,
    report,
    summary,
    monthly=None,
    categories=None,
    products=None,
    forecast=None,
):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output, engine="openpyxl"
    ) as writer:
        cleaned_df.to_excel(
            writer, sheet_name="Cleaned Data", index=False
        )

        report.to_excel(
            writer, sheet_name="Quality Report", index=False
        )

        pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"],
        ).to_excel(
            writer, sheet_name="Summary", index=False
        )

        if monthly is not None:
            monthly.to_excel(
                writer, sheet_name="Monthly Sales", index=False
            )

        if categories is not None:
            categories.to_excel(
                writer, sheet_name="Category Analysis", index=False
            )

        if products is not None:
            products.to_excel(
                writer, sheet_name="Top Products", index=False
            )

        if forecast is not None:
            forecast.to_excel(
                writer, sheet_name="Forecast", index=False
            )

    output.seek(0)
    return output.getvalue()


# ============================================================
# 8. SQLITE EXPORT
# ============================================================
def make_sqlite_bytes(dataframe):
    connection = sqlite3.connect(":memory:")

    sql_df = dataframe.copy()

    for col in sql_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sql_df[col]):
            sql_df[col] = sql_df[col].astype("string")

        elif pd.api.types.is_string_dtype(sql_df[col].dtype):
            sql_df[col] = sql_df[col].astype(object).where(
                sql_df[col].notna(), None
            )

    sql_df.to_sql(
        "cleaned_data",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.commit()

    try:
        database_bytes = connection.serialize()
    finally:
        connection.close()

    return database_bytes


# ============================================================
# 9. SALES ANALYTICS AND FORECAST
# ============================================================
def make_sales_analysis(
    df,
    date_col,
    sales_col,
    category_col=None,
    product_col=None,
):
    data = df.copy()

    data[date_col] = pd.to_datetime(
        data[date_col], errors="coerce", dayfirst=True
    )

    data[sales_col] = pd.to_numeric(
        data[sales_col], errors="coerce"
    )

    valid = data.dropna(
        subset=[date_col, sales_col]
    ).copy()

    if valid.empty:
        raise ValueError(
            "No valid date and sales values were found."
        )

    valid["Year-Month"] = (
        valid[date_col].dt.to_period("M").astype(str)
    )

    monthly = (
        valid.groupby("Year-Month", as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: "Sales"})
        .sort_values("Year-Month")
        .reset_index(drop=True)
    )

    categories = None

    if (
        category_col
        and category_col in valid.columns
        and category_col not in (date_col, sales_col)
    ):
        categories = (
            valid.groupby(category_col, dropna=False)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index(name="Sales")
        )

    products = None

    if (
        product_col
        and product_col in valid.columns
        and product_col not in (date_col, sales_col)
    ):
        products = (
            valid.groupby(product_col, dropna=False)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="Sales")
        )

    forecast = None

    if len(monthly) >= 3:
        monthly["Month_Number"] = np.arange(
            1, len(monthly) + 1
        )

        model = LinearRegression()

        model.fit(
            monthly[["Month_Number"]],
            monthly["Sales"],
        )

        next_number = len(monthly) + 1

        predicted = float(
            model.predict(
                pd.DataFrame({
                    "Month_Number": [next_number]
                })
            )[0]
        )

        last_period = pd.Period(
            monthly["Year-Month"].iloc[-1],
            freq="M",
        )

        forecast = pd.DataFrame({
            "Year-Month": [str(last_period + 1)],
            "Predicted Sales": [max(0.0, predicted)],
            "Method": ["Simple Linear Regression"],
        })

    kpis = {
        "Total Sales": float(valid[sales_col].sum()),
        "Valid Sales Rows": int(len(valid)),
        "Average Sales per Row": float(
            valid[sales_col].mean()
        ),
        "Months Analysed": int(len(monthly)),
    }

    return kpis, monthly, categories, products, forecast


# ============================================================
# 10. SESSION STATE
# ============================================================
def reset_results():
    keys = [
        "cleaned_df",
        "quality_report",
        "quality_summary",
        "invalid_counts",
        "type_errors",
        "duplicates_removed",
        "sales_kpis",
        "sales_monthly",
        "sales_categories",
        "sales_products",
        "sales_forecast",
        "sales_error",
    ]

    for key in keys:
        st.session_state.pop(key, None)

    st.session_state["cleaning_done"] = False


if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "type_settings" not in st.session_state:
    st.session_state.type_settings = {}

if "types_submitted" not in st.session_state:
    st.session_state.types_submitted = False

if "cleaning_done" not in st.session_state:
    st.session_state.cleaning_done = False


# ============================================================
# 11. UPLOAD DATASET
# ============================================================
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()

current_signature = hashlib.sha256(
    file_bytes
).hexdigest()

if st.session_state.file_signature != current_signature:
    try:
        uploaded_file.seek(0)
        loaded_df = read_uploaded_file(uploaded_file)

    except Exception as exc:
        st.error(f"Unable to read this file: {exc}")
        st.stop()

    if loaded_df.empty:
        st.error("The uploaded dataset has no rows.")
        st.stop()

    if len(loaded_df.columns) == 0:
        st.error("No columns were found in this file.")
        st.stop()

    loaded_df.columns = [
        str(col).strip() for col in loaded_df.columns
    ]

    if loaded_df.columns.duplicated().any():
        st.error(
            "Some column names become duplicates after trimming spaces. "
            "Rename them in the source file and upload again."
        )
        st.stop()

    st.session_state.file_signature = current_signature

    st.session_state.source_df = loaded_df.copy(deep=True)

    st.session_state.type_settings = {
        col: infer_column_type(loaded_df[col])
        for col in loaded_df.columns
    }

    st.session_state.types_submitted = False

    reset_results()
TYPE_OPTIONS = ["Text", "Whole Number", "Decimal Number", "Date"]
source_df = st.session_state.source_df.copy(deep=True)

st.success(
    "Dataset uploaded successfully. The original upload remains unchanged."
)

st.subheader("1. Original Dataset Preview")

m1, m2, m3 = st.columns(3)

m1.metric("Rows", f"{len(source_df):,}")
m2.metric("Columns", f"{len(source_df.columns):,}")
m3.metric(
    "Duplicate Rows",
    f"{int(source_df.duplicated().sum()):,}",
)

st.dataframe(source_df.head(30), width="stretch")

with st.expander("View complete original dataset"):
    st.dataframe(source_df, width="stretch")


# ============================================================
# 12. REVIEW COLUMN TYPES
# ============================================================
st.subheader("2. Review Column Types")

st.write(
    "Review the suggested type for every column. "
    "Keep IDs, phone numbers and postal codes as Text "
    "if leading zeros matter."
)

selected_types = {}

with st.form("column_type_form"):
    for index, col in enumerate(source_df.columns):
        suggestion = infer_column_type(source_df[col])

        current = st.session_state.type_settings.get(
            col, suggestion
        )

        selected_types[col] = st.selectbox(
            f"{col} — suggested: {suggestion}",
            options=TYPE_OPTIONS,
            index=(
                TYPE_OPTIONS.index(current)
                if current in TYPE_OPTIONS
                else TYPE_OPTIONS.index(suggestion)
            ),
            key=f"type_select_{index}_{str(col)}",
        )

    submitted_types = st.form_submit_button(
        "Apply Column Type Settings",
        type="primary",
    )

if submitted_types:
    st.session_state.type_settings = selected_types.copy()
    st.session_state.types_submitted = True

    reset_results()

    st.success(
        "Column type settings saved. Cleaning options are now available."
    )


# ============================================================
# 13. DATA CLEANING
# ============================================================
if not st.session_state.types_submitted:
    st.info(
        "Review the column types and click "
        "'Apply Column Type Settings' to continue."
    )

else:
    type_settings = st.session_state.type_settings

    st.subheader("3. Data Cleaning")

    st.write(
        "Cleaning creates a separate dataset. "
        "Your original upload remains unchanged."
    )

    with st.form("cleaning_rules_form"):
        replace_numeric_nulls = st.checkbox(
            "Replace missing numeric values",
            value=True,
        )

        numeric_replacement = st.number_input(
            "Numeric null replacement value",
            value=1.0,
            step=1.0,
            disabled=not replace_numeric_nulls,
        )

        replace_text_nulls = st.checkbox(
            "Replace missing text values with 'Unknown'",
            value=False,
        )

        remove_duplicates = st.checkbox(
            "Remove exact duplicate rows",
            value=False,
        )

        confirm_duplicate_removal = False

        if remove_duplicates:
            confirm_duplicate_removal = st.checkbox(
                "I confirm removal of exact duplicate rows",
                value=False,
            )

        clean_clicked = st.form_submit_button(
            "🧹 Clean Dataset",
            type="primary",
        )

    if clean_clicked:
        cleaned_df = source_df.copy(deep=True)

        invalid_counts = {}
        type_errors = []

        replacement = (
            numeric_replacement
            if replace_numeric_nulls
            else None
        )

        for col in source_df.columns:
            try:
                converted, invalid_count = convert_column(
                    source_df[col],
                    type_settings[col],
                    replacement,
                )

                cleaned_df[col] = converted
                invalid_counts[col] = invalid_count

            except Exception as exc:
                type_errors.append(f"{col}: {exc}")
                cleaned_df[col] = source_df[col].copy()
                invalid_counts[col] = 0

        if replace_text_nulls:
            for col in cleaned_df.columns:
                if type_settings.get(col) == "Text":
                    cleaned_df[col] = cleaned_df[col].fillna(
                        "Unknown"
                    )

        duplicates_removed = 0

        if remove_duplicates and confirm_duplicate_removal:
            before = len(cleaned_df)

            cleaned_df = cleaned_df.drop_duplicates().copy()

            duplicates_removed = before - len(cleaned_df)

        elif remove_duplicates:
            st.warning(
                "Duplicate removal was not confirmed, "
                "so no rows were removed."
            )

        report, summary = build_quality_report(
            source_df,
            cleaned_df,
            invalid_counts,
            duplicates_removed,
        )

        st.session_state.cleaned_df = cleaned_df
        st.session_state.quality_report = report
        st.session_state.quality_summary = summary
        st.session_state.invalid_counts = invalid_counts
        st.session_state.type_errors = type_errors
        st.session_state.duplicates_removed = duplicates_removed
        st.session_state.cleaning_done = True

        for key in (
            "sales_kpis",
            "sales_monthly",
            "sales_categories",
            "sales_products",
            "sales_forecast",
            "sales_error",
        ):
            st.session_state.pop(key, None)


# ============================================================
# 14. VALIDATION AND ORIGINAL VS CLEANED REPORT
# ============================================================
if st.session_state.cleaning_done:
    cleaned_df = st.session_state.cleaned_df
    quality_report = st.session_state.quality_report
    summary = st.session_state.quality_summary
    invalid_counts = st.session_state.invalid_counts

    st.subheader("4. Validation & Original vs Cleaned Comparison")

    a, b, c, d = st.columns(4)

    a.metric("Rows Before", f"{len(source_df):,}")
    b.metric("Rows After", f"{len(cleaned_df):,}")

    c.metric(
        "Null Cells Before",
        f"{summary['Null Cells Before']:,}",
    )

    d.metric(
        "Null Cells After",
        f"{summary['Null Cells After']:,}",
    )

    st.write("Dataset Summary")

    st.dataframe(
        pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"],
        ),
        width="stretch",
    )

    st.write("Column-Level Quality Report")
    st.dataframe(quality_report, width="stretch")

    if st.session_state.type_errors:
        st.warning(
            "Some columns could not be converted. "
            "Original values were retained for those columns."
        )

        for message in st.session_state.type_errors:
            st.write(f"- {message}")

    if sum(invalid_counts.values()) > 0:
        st.warning(
            "Invalid non-empty values were detected during conversion. "
            "Review the quality report before using the data."
        )

    st.write("Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(50), width="stretch")

    with st.expander("View complete cleaned dataset"):
        st.dataframe(cleaned_df, width="stretch")


    # ========================================================
    # 15. OPTIONAL SALES ANALYTICS AND FORECASTING
    # ========================================================
    st.subheader("5. Sales Analytics & Forecasting (Optional)")

    date_guess = find_matching_column(
        cleaned_df,
        [
            "order date",
            "date",
            "transaction date",
            "invoice date",
            "purchase date",
            "sales date",
        ],
    )

    sales_guess = find_matching_column(
        cleaned_df,
        [
            "sales",
            "revenue",
            "total sales",
            "sales amount",
            "amount",
            "total amount",
            "order value",
        ],
    )

    category_guess = find_matching_column(
        cleaned_df,
        ["category", "product category", "department"],
    )

    product_guess = find_matching_column(
        cleaned_df,
        ["product name", "product", "item", "item name"],
    )

    if date_guess is not None and sales_guess is not None:
        st.info(
            "Potential date and sales columns were detected. "
            "Confirm the selections before running analytics."
        )

        all_columns = list(cleaned_df.columns)

        date_col = st.selectbox(
            "Sales date column",
            all_columns,
            index=all_columns.index(date_guess),
            key="analytics_date_column",
        )

        sales_col = st.selectbox(
            "Sales/revenue column",
            all_columns,
            index=all_columns.index(sales_guess),
            key="analytics_sales_column",
        )

        optional_columns = ["(Not available)"] + all_columns

        category_index = (
            optional_columns.index(category_guess)
            if category_guess in optional_columns
            else 0
        )

        product_index = (
            optional_columns.index(product_guess)
            if product_guess in optional_columns
            else 0
        )

        category_col = st.selectbox(
            "Category column",
            optional_columns,
            index=category_index,
            key="analytics_category_column",
        )

        product_col = st.selectbox(
            "Product column",
            optional_columns,
            index=product_index,
            key="analytics_product_column",
        )

        if st.button(
            "Run Sales Analysis & Forecast",
            type="primary",
            key="run_sales_analysis",
        ):
            try:
                (
                    kpis,
                    monthly,
                    categories,
                    products,
                    forecast,
                ) = make_sales_analysis(
                    cleaned_df,
                    date_col,
                    sales_col,
                    None if category_col == "(Not available)" else category_col,
                    None if product_col == "(Not available)" else product_col,
                )

                st.session_state.sales_kpis = kpis
                st.session_state.sales_monthly = monthly
                st.session_state.sales_categories = categories
                st.session_state.sales_products = products
                st.session_state.sales_forecast = forecast

                st.session_state.pop("sales_error", None)

            except Exception as exc:
                st.session_state.sales_error = str(exc)

                for key in (
                    "sales_kpis",
                    "sales_monthly",
                    "sales_categories",
                    "sales_products",
                    "sales_forecast",
                ):
                    st.session_state.pop(key, None)

        if st.session_state.get("sales_error"):
            st.error(st.session_state.sales_error)

        if st.session_state.get("sales_kpis"):
            kpis = st.session_state.sales_kpis

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Total Sales",
                f"{kpis['Total Sales']:,.2f}",
            )

            k2.metric(
                "Valid Sales Rows",
                f"{kpis['Valid Sales Rows']:,}",
            )

            k3.metric(
                "Average Sales per Row",
                f"{kpis['Average Sales per Row']:,.2f}",
            )

            k4.metric(
                "Months Analysed",
                f"{kpis['Months Analysed']:,}",
            )

            monthly = st.session_state.sales_monthly

            st.markdown("#### Monthly Sales Trend")

            st.line_chart(
                monthly.set_index("Year-Month")["Sales"]
            )

            st.dataframe(monthly, width="stretch")

            categories = st.session_state.sales_categories

            if categories is not None and not categories.empty:
                st.markdown("#### Sales by Category")

                st.bar_chart(
                    categories.set_index(
                        categories.columns[0]
                    )["Sales"]
                )

                st.dataframe(categories, width="stretch")

            products = st.session_state.sales_products

            if products is not None and not products.empty:
                st.markdown("#### Top 10 Products by Sales")

                st.bar_chart(
                    products.set_index(
                        products.columns[0]
                    )["Sales"]
                )

                st.dataframe(products, width="stretch")

            forecast = st.session_state.sales_forecast

            if forecast is not None:
                st.markdown("#### Next-Month Regression Forecast")

                st.metric(
                    f"Predicted Sales for {forecast['Year-Month'].iloc[0]}",
                    f"{forecast['Predicted Sales'].iloc[0]:,.2f}",
                )

                st.dataframe(forecast, width="stretch")

                st.caption(
                    "This is a simple linear-regression estimate "
                    "based on monthly totals. It is not a guarantee "
                    "and does not model seasonality."
                )

            else:
                st.info(
                    "At least 3 months of valid sales data are needed "
                    "for the simple regression forecast."
                )

    else:
        st.info(
            "Sales analytics is hidden because matching date and sales "
            "columns were not detected. Cleaning and exports are still "
            "available for any dataset."
        )


    # ========================================================
    # 16. EXCEL EXPORT
    # ========================================================
    st.subheader("6. Excel Export")

    excel_bytes = make_excel_file(
        cleaned_df,
        quality_report,
        summary,
        monthly=st.session_state.get("sales_monthly"),
        categories=st.session_state.get("sales_categories"),
        products=st.session_state.get("sales_products"),
        forecast=st.session_state.get("sales_forecast"),
    )

    st.download_button(
        label="📥 Download Cleaned Excel Report",
        data=excel_bytes,
        file_name="cleaned_dataset_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )


    # ========================================================
    # 17. SQLITE DATABASE EXPORT
    # ========================================================
    st.subheader("7. SQLite Export")

    try:
        sqlite_bytes = make_sqlite_bytes(cleaned_df)

        st.download_button(
            label="📥 Download SQLite Database (.db)",
            data=sqlite_bytes,
            file_name="cleaned_dataset.db",
            mime="application/octet-stream",
            width="stretch",
        )

    except Exception as exc:
        st.error(f"Could not create SQLite database: {exc}")

    st.success(
        "Cleaning, validation and export results are ready. "
        "Review the report before using the cleaned data."
    )


st.divider()

st.caption(
    "Universal Data Cleaning & Analytics Platform | "
    "Original data is preserved separately."
)
