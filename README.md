After mapping the original mental-health labels to anxiety levels (Low, Moderate, High), the dataset was validated to ensure that the mapping was applied correctly.

Two checks were performed:

1. Checking for missing values in the anxiety_level column using:
df['anxiety_level'].isnull().sum()

2. Checking the distribution of anxiety levels using:
df['anxiety_level'].value_counts()

The results confirmed that there were no missing values in the anxiety_level column and all labels were successfully mapped. The distribution of anxiety levels across the dataset was also verified, ensuring the dataset is consistent and ready for further preprocessing and model training.# AI Exam Anxiety Detector 
