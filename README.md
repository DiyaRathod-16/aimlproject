# JobGuard - Fake Job Posting Detection

A Streamlit AIML project for training a fake-job-posting classifier from a labeled CSV, exploring the data, evaluating the model, and scoring new postings.

## Project layout

```text
fake-job-posting-detection/
├── .streamlit/
│   └── config.toml
├── app.py
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

The source dataset is not copied into the app repository. Upload `fake_job_postings.csv` in the app, or provide another labeled dataset in the same format.

## Run locally

Use Python 3.10 or newer.

```bash
cd fake-job-posting-detection
python -m venv .venv
```

Activate the environment, then install and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `\.venv\Scripts\Activate.ps1`.

## Deploy on Streamlit Community Cloud

1. Push this project directory to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the app's main file path to `app.py` (or `fake-job-posting-detection/app.py` if the project is nested in a larger repository).
4. Deploy. Streamlit installs the packages listed in `requirements.txt`.
5. Upload the labeled CSV in the running app.

## Dataset format

Required: one target/label column with two classes and at least one text field. For the supplied Fake Job Postings dataset, choose `fraudulent` (`0` = genuine, `1` = fake). The app recognizes these common text fields when present: `title`, `company_profile`, `description`, `requirements`, and `benefits`. You can change the selected text fields before training.

For a second, unlabeled CSV, include the same text-field columns selected during training. Extra columns are preserved in the downloadable prediction file.

## What the app shows

- Dataset dimensions, class balance, missing-value chart, preview, and numeric correlation heatmap.
- TF-IDF text features and class-balanced logistic regression, matching the approach in the project PDF.
- Stratified 80/20 train/test evaluation. TF-IDF is fitted only on the training split to avoid leaking test vocabulary into training.
- Accuracy, balanced accuracy, fake-class precision/recall/F1, classification report, ROC curve, and an interactive confusion-matrix heatmap.
- A two-component PCA scatter plot of engineered text-length, word-count, and available listing-flag features. It is a dataset exploration view, not the classifier's decision surface.
- Single-posting scoring and batch CSV scoring with a downloadable result.

The PCA projection is limited to 3,500 randomly selected rows when a larger dataset is uploaded, to keep the chart responsive. A model score is a screening signal, not proof that a posting or employer is fraudulent.
