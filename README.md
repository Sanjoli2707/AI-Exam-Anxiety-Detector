After completing the preprocessing and label mapping steps, the final dataset was prepared for machine learning model training.

The processed dataset now contains:
- Cleaned text statements
- Anxiety levels mapped to numerical labels
- Reduced label complexity for better model learning

The dataset was verified to ensure that the text data and corresponding anxiety labels were correctly aligned. The final dataset structure includes the text input and the corresponding numerical anxiety label, which will be used as the target variable during model training.

This processed dataset is now ready to be used for training the anxiety detection model.
After mapping the original mental-health labels to anxiety levels (Low, Moderate, High), the dataset was validated to ensure that the mapping was applied correctly.

Two checks were performed:

1. Checking for missing values in the anxiety_level column using:
df['anxiety_level'].isnull().sum()

2. Checking the distribution of anxiety levels using:
df['anxiety_level'].value_counts()

The results confirmed that there were no missing values in the anxiety_level column and all labels were successfully mapped. The distribution of anxiety levels across the dataset was also verified, ensuring the dataset is consistent and ready for further preprocessing and model training.# AI Exam Anxiety Detector 
## Activity 3.4 – Creating Numerical Labels

Machine learning models require numerical labels rather than text labels.

The anxiety levels were encoded as:

0 → Low Anxiety  
1 → Moderate Anxiety  
2 → High Anxiety  

Implementation:

```python
numeric_mapping = {
'Low': 0,
'Moderate': 1,
'High': 2
}

df['anxiety_label'] = df['anxiety_level'].map(numeric_mapping)
## Activity 3.3 – Anxiety Level Label Mapping

Original mental health labels were mapped into three anxiety levels to align with the project objective.

Label Mapping:
- Normal → Low Anxiety
- Anxiety → Moderate Anxiety
- Depression → High Anxiety
- Suicidal → High Anxiety

This mapping simplifies the classification problem and focuses the model specifically on predicting exam-related anxiety levels.
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
## Activity 3.1 – Handling Missing Text Data

Missing values in the text column were checked using the pandas function:

df['text'].isnull().sum()

Result:
- text → 0 missing values

Since no missing values were found in the text column, no data cleaning or imputation was required. The dataset is ready for further preprocessing steps.
## Activity 2.6 – Dataset Suitability Assessment

Based on the exploratory analysis of the dataset, the following observations were made:

- The dataset contains a large number of samples, making it suitable for training deep learning models.
- The text entries contain emotional expressions that can help identify anxiety-related patterns.
- Multiple mental health categories are present, allowing meaningful classification and grouping.

Therefore, the dataset is considered suitable for building an AI-based exam anxiety detection system using Natural Language Processing (NLP) techniques.
<img width="839" height="398" alt="image" src="https://github.com/user-attachments/assets/a2f65870-0e89-4978-a30e-4cb98b7a335b" />## Activity 2.5 – Class Distribution Analysis

Class distribution was analyzed using the pandas function `df['status'].value_counts()`.

Results:
- Normal → 18391
- Depression → 14506
- Suicidal → 11212
- Anxiety → 5503

The analysis shows that the dataset contains multiple mental health categories with different frequencies. 
This indicates the presence of class imbalance, where some categories have significantly more samples than others.

Understanding class distribution helps prevent model bias and guides preprocessing strategies for training machine learning models.
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
