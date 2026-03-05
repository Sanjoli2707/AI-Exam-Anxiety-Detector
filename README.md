## Activity 3.2 – Understanding Original Labels

The original labels in the dataset were examined using the pandas function:

df['status'].unique()

Result:
The dataset contains the following mental health categories:
- Normal
- Depression
- Suicidal
- Anxiety

These labels represent different mental health conditions. However, the objective of the project is to classify exam anxiety levels rather than clinical diagnoses. Therefore, these labels will later be mapped into three anxiety levels: Low, Moderate, and High.
