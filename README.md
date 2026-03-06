## Activity 2.4 – Checking Missing Values

Missing values in the dataset were checked using the pandas function `df.isnull().sum()`.

Results:
- Unique_ID → 9600 missing values
- text → 0 missing values
- status → 0 missing values

The analysis shows that missing values exist only in the `Unique_ID` column, which is not required for model training. The important columns (`text` and `status`) do not contain missing values, so the dataset is suitable for further preprocessing and model training.
# AI Exam Anxiety Detector 

## Activity 2.3 – Understanding Dataset Structure

The dataset was analyzed using pandas functions such as `df.head()`, `df.shape`, `df.columns`, and `df.dtypes`.

Dataset Details:
- Total Records: 49,612
- Total Columns: 3

Columns:
1. Unique_ID – Identifier for each record
2. text – Student written statement
3. status – Mental health label

Data Types:
- Unique_ID → float64
- text → object
- status → object

This dataset structure confirms that the `text` column can be used as input for Natural Language Processing models such as BERT for anxiety detection.
