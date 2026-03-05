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
