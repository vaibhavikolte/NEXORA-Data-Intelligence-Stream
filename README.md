# NEXORA – Data Intelligence Stream

**An Interactive Data Cleaning and Analytics Application**

NEXORA – Data Intelligence Stream is a Python-based data analytics application designed to simplify data cleaning, data quality assessment, exploratory analysis, and sales forecasting through an interactive web interface.

The application allows users to upload datasets, review data quality, clean common data issues, explore analytical insights, and export processed data for further analysis.

## 🚀 Key Features

- **Dataset Upload:** Upload CSV, Excel (`.xlsx`), and supported Excel (`.xls`) files.
- **Data Preview:** Preview uploaded datasets and review column data types.
- **Data Cleaning:** Handle missing values, review invalid values, and optionally remove duplicate records.
- **Data Quality Analysis:** Compare data quality before and after cleaning.
- **Sales Analytics:** Analyze sales trends when suitable date and sales columns are available.
- **Monthly Sales Trends:** Examine monthly sales patterns.
- **Category Analysis:** Explore category-level performance when relevant columns exist.
- **Top Product Analysis:** Identify leading products based on available sales data.
- **Sales Forecasting:** Apply a basic Linear Regression model to estimate future sales when sufficient monthly historical data is available.
- **Data Export:** Export processed data to Excel and SQLite for further use.

## 🛠️ Technologies Used

- **Python** – Application logic and data processing
- **Streamlit** – Interactive web application interface
- **Pandas** – Data manipulation and cleaning
- **NumPy** – Numerical operations
- **Scikit-learn** – Linear Regression forecasting
- **OpenPyXL** – Excel file processing
- **SQLite** – Structured data storage and export

## ⚙️ How It Works

1. Upload a supported dataset.
2. Preview the data and inspect its structure.
3. Review data quality issues.
4. Apply the available cleaning operations.
5. Compare data quality before and after cleaning.
6. Explore sales analytics and forecasting when the required data is available.
7. Export the cleaned dataset for further analysis.

## 📁 Project Structure

```text
NEXORA-Data-Intelligence-Stream/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Project Objective

The objective of NEXORA is to make common data preparation and analytics tasks more accessible through a single interactive application. It combines data cleaning, quality assessment, analytical exploration, basic forecasting, and data export in one workflow.

## 👩‍💻 Author

**Vaibhavi Kolte**

Aspiring Data Analyst | Python | SQL | Power BI | Excel

- **GitHub:** [vaibhavikolte](https://github.com/vaibhavikolte)
- **LinkedIn:** [Vaibhavi Kolte](https://www.linkedin.com/in/vaibhavi-kolte-0824a0420/)

---

*Built as a data analytics project to practice data processing, data quality assessment, visualization, and basic predictive analysis.*
