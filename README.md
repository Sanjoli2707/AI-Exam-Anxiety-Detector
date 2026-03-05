## Activity 2.4 – Checking Missing Values

Missing values in the dataset were checked using the pandas function `df.isnull().sum()`.

Results:
- Unique_ID → 9600 missing values
- text → 0 missing values
- status → 0 missing values

The analysis shows that missing values exist only in the `Unique_ID` column, which is not required for model training. The important columns (`text` and `status`) do not contain missing values, so the dataset is suitable for further preprocessing and model training.
