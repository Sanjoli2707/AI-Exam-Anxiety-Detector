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
