# Fake Job Posting Detection: Project Study Guide

## 1. Project purpose

This project is a **binary text-classification problem**. Given the text of a job posting, the model predicts one of two labels:

| Label in dataset | Meaning |
| --- | --- |
| `0` | Genuine job posting |
| `1` | Fraudulent or fake job posting |

The practical aim is to identify postings that deserve further review. The output is a probability-like model score and a predicted class. It should be used as a screening aid because text alone cannot prove whether an employer or vacancy is genuine.

## 2. Dataset study

### Basic profile

The supplied `fake_job_postings.csv` contains **17,880 rows** and **18 columns**. It is an imbalanced dataset:

| Class | Count | Share |
| --- | ---: | ---: |
| Genuine (`0`) | 17,014 | 95.16% |
| Fake (`1`) | 866 | 4.84% |

This imbalance is important. A weak model that predicts every listing as genuine would still appear about 95% accurate, while detecting zero fake jobs. Therefore the project must report fake-class precision, recall and F1 score, not accuracy alone.

### Columns

| Column | Type / role | Description |
| --- | --- | --- |
| `job_id` | Identifier | Unique listing identifier. Do not use as model text: identifiers do not describe fraud. |
| `title` | Text | Job title. |
| `location` | Categorical text | Country, state and city where supplied. |
| `department` | Categorical text | Department or team. |
| `salary_range` | Categorical text | Salary range where supplied. |
| `company_profile` | Text | Employer description. |
| `description` | Text | Main job description. |
| `requirements` | Text | Required skills, qualifications and experience. |
| `benefits` | Text | Benefits and perks. |
| `telecommuting` | Binary numeric | Whether remote work is mentioned. |
| `has_company_logo` | Binary numeric | Whether a company logo is present. |
| `has_questions` | Binary numeric | Whether the listing asks application questions. |
| `employment_type` | Categorical | Full-time, part-time, contract and similar values. |
| `required_experience` | Categorical | Experience level. |
| `required_education` | Categorical | Education requirement. |
| `industry` | Categorical | Industry label. |
| `function` | Categorical | Job function. |
| `fraudulent` | Target | The label: 0 genuine, 1 fake. |

### Missing data

Several columns have substantial missing values. The important rates are:

| Column | Missing share |
| --- | ---: |
| `salary_range` | 83.96% |
| `department` | 64.58% |
| `required_education` | 45.33% |
| `benefits` | 40.34% |
| `required_experience` | 39.43% |
| `function` | 36.10% |
| `industry` | 27.42% |
| `employment_type` | 19.41% |
| `company_profile` | 18.50% |
| `requirements` | 15.08% |

For the model, missing text is safely replaced by an empty string before the selected fields are joined. This keeps each row without inventing text. The app also visualizes missingness so users can inspect the uploaded data quality.

### Useful exploratory findings

These are **associations in this particular labeled dataset**, not universal fraud rules:

- Listings without a company logo have a 15.93% fake rate, compared with 1.99% where a logo is present.
- Listings marked as telecommuting have an 8.34% fake rate, compared with 4.69% for other listings.
- Listings with application questions have a 2.84% fake rate, compared with 6.78% without questions.
- `function = Administrative` has an 18.89% fake rate among 630 rows. `US, TX, Houston` has a 34.20% fake rate among 269 rows.

Small groups can have unstable percentages, so never interpret a high rate for a rare category as a definite rule. Also, the application model primarily learns from text rather than treating these examples as fixed decision rules.

## 3. Data preparation in the app

The app lets the user choose the target and text columns. For the supplied dataset, select:

- Target: `fraudulent`
- Text fields: `title`, `company_profile`, `description`, `requirements`, `benefits`

The selected text fields are concatenated into one document per posting. Empty cells become empty text. Text is normalized by collapsing repeated whitespace.

The records are split using an **80/20 stratified train-test split** with `random_state=42`.

- 80% is used to train the model.
- 20% is held back for final evaluation.
- Stratification keeps the fake/genuine proportion similar in both sets.
- `random_state=42` makes the split repeatable.

## 4. Machine-learning concepts used

### Supervised learning

The dataset already contains the correct answer (`fraudulent`) for each row. The model learns a mapping from input text to that known label. This is supervised learning.

### Binary classification

There are exactly two outcomes: genuine or fake. The classifier estimates the probability that a posting belongs to the fake class and uses a threshold to turn that score into a label.

### Train-test split and generalization

The aim is to perform well on new postings, not only on postings seen during training. The test split imitates unseen future data. Reporting results on the same rows used for training would overestimate the model's real-world usefulness.

### TF-IDF vectorization

Machine-learning algorithms need numbers, but a job description is text. **TF-IDF** converts words and phrases into numeric features.

- **Term frequency (TF):** a word receives more importance when it appears more often in a particular posting.
- **Inverse document frequency (IDF):** a word that occurs in nearly every posting receives less importance because it is less useful for distinguishing classes.
- The output is a sparse matrix: each row is a posting and each column is a word or phrase feature.

The app uses up to 3,000 features, English stop-word removal, and one-word and two-word phrases (unigrams and bigrams). Examples of stop words include common words such as “the” and “and”, which usually carry little classification value.

### Leakage prevention

**Data leakage** occurs when information from the test data enters the training process. It makes evaluation unrealistically optimistic. The supplied PDF fits TF-IDF before splitting. The Streamlit app improves that sequence: it first splits the raw text, then fits TF-IDF only on the training split. Test postings are transformed using the vocabulary already learned from training.

### Logistic regression

Despite its name, logistic regression is a classification algorithm. It calculates a weighted combination of TF-IDF features, then passes it through the logistic (sigmoid) function:

`P(fake) = 1 / (1 + e^(-z))`

Here, `z` is the weighted sum of text features plus an intercept. A positive word weight pushes the score toward fake; a negative weight pushes it toward genuine. The app displays the strongest learned text signals to make this more interpretable.

### Class weighting

The fake class is rare. The app sets `class_weight='balanced'`, which makes mistakes on fake examples cost more during training. Without this, the model could minimize overall error by mostly predicting genuine.

### Classification threshold

The default threshold is 0.50: predict fake when the estimated fake score is at least 50%. The app allows a user to adjust the threshold.

- Lowering the threshold catches more fake listings (higher recall) but can create more false alarms.
- Raising the threshold makes a fake label harder to obtain (often higher precision) but can miss more fake listings.

## 5. Evaluation concepts

For this project, “positive” means a fake listing.

| Outcome | Meaning |
| --- | --- |
| True positive (TP) | Fake posting predicted as fake. |
| False positive (FP) | Genuine posting predicted as fake. |
| True negative (TN) | Genuine posting predicted as genuine. |
| False negative (FN) | Fake posting predicted as genuine. |

### Confusion matrix

The confusion matrix is a 2×2 table containing TN, FP, FN and TP. The app renders it as a heatmap. It shows the actual count of correct and incorrect outcomes at the currently selected threshold.

### Accuracy

`Accuracy = (TP + TN) / all predictions`

Accuracy is easy to understand but misleading on an imbalanced dataset. Use it together with the metrics below.

### Precision

`Precision = TP / (TP + FP)`

Fake precision answers: “When the model says fake, how often is it correct?” High precision reduces unnecessary false alarms.

### Recall (sensitivity)

`Recall = TP / (TP + FN)`

Fake recall answers: “Of all fake listings, how many did the model catch?” High recall reduces missed fake listings.

### F1 score

`F1 = 2 × (precision × recall) / (precision + recall)`

F1 combines precision and recall. It is particularly useful when one class is much rarer than the other.

### Balanced accuracy

Balanced accuracy averages recall for both classes. It gives genuine and fake examples equal importance despite the class imbalance.

### ROC curve and AUC

The ROC curve plots true-positive rate against false-positive rate over many thresholds. **AUC** is the area under that curve:

- 0.5: no better than random ranking.
- 1.0: perfect separation on the test set.

A high AUC means the model tends to rank fake postings above genuine postings. It does not decide which operating threshold is best.

## 6. PCA visualization

**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction technique. It transforms many numerical inputs into a few new components that retain as much variation as possible.

The app's PCA chart uses engineered, interpretable numerical features: word and character counts for text fields, available binary listing flags, and whether a location was provided. It standardizes them, then plots the first two principal components.

Use the PCA plot to inspect clustering, overlap and unusual points. It is not the classifier's decision boundary and does not use the TF-IDF text matrix directly; therefore it should not be used alone to claim that the model separates fake and genuine postings.

## 7. Heatmaps and visualizations in the app

| Visualization | Question it answers |
| --- | --- |
| Class-balance donut | How imbalanced are genuine and fake labels? |
| Missing-value bar chart | Which columns have incomplete information? |
| Correlation heatmap | Which numeric listing flags move together? Correlation does not show causation. |
| Confusion-matrix heatmap | What kinds of prediction mistakes did the model make? |
| ROC curve | How does fake detection trade off against false alarms across thresholds? |
| PCA scatter plot | Do engineered posting characteristics show visible structure? |
| Top-terms chart | Which learned terms push model scores toward fake or genuine? |

## 8. Reference result from the supplied project PDF

The provided PDF describes a similar TF-IDF and class-balanced logistic regression experiment using `title`, `company_profile`, and `description`. It reports 96.22% accuracy on a 3,576-row test split. For the fake class, it reports 57% precision, 90% recall and 0.70 F1, with confusion matrix `[[3285, 118], [17, 156]]`.

Use those values as a reference for the PDF experiment only. Your Streamlit result may differ because it can use five text fields, uses word pairs, and prevents TF-IDF vocabulary leakage. The values shown inside your running app are the results to present for the deployed implementation.

## 9. Limitations and responsible interpretation

- Labels may contain errors, and the source of every posting has not been independently verified by this app.
- The dataset may represent particular locations, time periods, industries and language patterns, so performance can change on new sources.
- A model can learn wording patterns that correlate with the label without identifying actual fraud.
- Empty or short postings provide less text evidence.
- A probability is a model score, not a verified fact.

For a real screening workflow, a high-risk prediction should trigger manual checks: employer website, official email domain, salary and payment requests, job-board history, and independent company verification.

## 10. Viva / presentation questions and short answers

**Why did you choose logistic regression?** It is fast for sparse TF-IDF text, gives class probabilities, and is interpretable through learned feature weights.

**Why not rely only on accuracy?** Only 4.84% of this dataset is fake. Predicting every row as genuine would have high accuracy but catch no fake jobs.

**Why use class weights?** The fake class has only 866 rows. Balancing weights tells the algorithm to give rare fake examples more importance.

**Why stratify the split?** It preserves the genuine/fake ratio in both train and test data, which makes the test set representative.

**What is TF-IDF?** It represents each job posting numerically, increasing the importance of terms that are frequent in one posting but uncommon across all postings.

**What is data leakage?** It is when information from the test set influences training. Fitting TF-IDF before splitting is a leakage risk, so the app fits it after the split using training data only.

**What is PCA used for here?** Exploration. It reduces engineered posting features to two dimensions so patterns and outliers can be viewed. It is not used to make the final fake/genuine prediction.

**What is the key limitation?** The model detects patterns from historical labels, not verified truth about a live employer. It supports human review rather than replacing it.
