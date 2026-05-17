# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn (version 1.5.1). It uses 100 decision tree estimators with a fixed random seed of 42 for reproducibility. The model was developed as part of the Udacity Machine Learning DevOps Engineer Nanodegree (D501 project) to demonstrate a scalable ML pipeline deployed with FastAPI.

## Intended Use

The model is intended to predict whether a person's annual income exceeds $50,000 based on demographic and employment features from US Census data. It is designed for educational use to demonstrate MLOps best practices including model training, slice-based evaluation, RESTful API deployment, and continuous integration.

## Training Data

The model was trained on the UCI Adult Census Income dataset (also known as the "Adult" dataset). The full dataset contains 32,561 records. After removing rows with missing values (marked as "?"), approximately 30,162 records remain. The data was split 80/20 into training and test sets using a random seed of 42. The training set contains approximately 24,129 samples.

The target label is the `salary` column, which takes the value `>50K` or `<=50K`. Categorical features were encoded using scikit-learn's `OneHotEncoder` and the label was binarized using `LabelBinarizer`.

The categorical features used are: workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Evaluation Data

The evaluation set consists of the remaining 20% of the cleaned dataset, approximately 6,033 samples. The same `OneHotEncoder` and `LabelBinarizer` fitted on the training data are applied to the test set in inference mode to prevent data leakage.

## Metrics

The model is evaluated using precision, recall, and F1 score (beta=1). Overall performance on the held-out test set:

- **Precision: 0.7419**
- **Recall: 0.6384**
- **F1: 0.6863**

Slice-level performance is computed for each unique value of each categorical feature and written to `slice_output.txt`. A selection of slice results is shown below:

- workclass: Private, Count: 4,536 — Precision: 0.7406 | Recall: 0.6427 | F1: 0.6882
- education: Bachelors, Count: 1,053 — Precision: 0.7837 | Recall: 0.7500 | F1: 0.7665
- sex: Male, Count: 4,026 — Precision: 0.7429 | Recall: 0.6736 | F1: 0.7066
- sex: Female, Count: 2,007 — Precision: 0.7362 | Recall: 0.5385 | F1: 0.6224

## Ethical Considerations

This dataset and model contain demographic features such as race, sex, and native-country, which are sensitive attributes. The model exhibits performance disparities across these slices. For example, the model shows notably lower recall for female individuals compared to male individuals. This model should not be used in any real-world decision-making context related to employment, lending, or benefits eligibility. It is intended solely for educational and demonstrative purposes.

Users should be aware that historical census data encodes societal biases and that a model trained on such data may perpetuate those biases. Slice-based evaluation is an important first step in identifying these disparities.

## Caveats and Recommendations

The model was not tuned extensively. Hyperparameter search (e.g., grid search or random search) could improve overall performance. The dataset is from the 1994 US Census and may not reflect current income distributions. Features such as `fnlgt` (final weight) are included as continuous inputs without scaling, which may affect model behaviour. It is recommended to apply feature scaling for models sensitive to input magnitude. Future work could include fairness-aware training, more extensive cross-validation, and deployment monitoring.
