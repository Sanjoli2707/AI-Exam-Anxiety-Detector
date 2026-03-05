## Activity 3.1 – Handling Missing Text Data

Missing values in the text column were checked using the pandas function:

df['text'].isnull().sum()

Result:
- text → 0 missing values

Since no missing values were found in the text column, no data cleaning or imputation was required. The dataset is ready for further preprocessing steps.
