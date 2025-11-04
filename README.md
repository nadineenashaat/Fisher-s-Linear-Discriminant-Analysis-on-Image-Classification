 Project Summary

This project applies Fisher’s Linear Discriminant Analysis (LDA) to classify handwritten digit images. The goal is to distinguish between 10 digit classes (0–9) using statistical separation based on class and non-class scatter matrices.

---

 Methodology

- **Data Loading**: 2400 training images and 200 test images were flattened into 1D vectors.
- **Class Separation**: Training data was split into 10 classes and their complements.
- **Mean Calculation**: Computed mean vectors for each class and non-class group.
- **Scatter Matrices**: Calculated within-class and non-class scatter matrices.
- **Fisher Weights**: Derived optimal weights using inverse scatter matrices.
- **Prediction**: Classified test images by minimizing Fisher scores.
- **Evaluation**: Used confusion matrix and accuracy score to assess performance.

---

 Results

- **Accuracy**: Achieved high classification accuracy using Fisher’s method.
- **Confusion Matrix**: Saved as `ConfusionNoBias.jpg` for visual inspection.
- **Bias Terms**: Calculated for each class to define decision boundaries.
