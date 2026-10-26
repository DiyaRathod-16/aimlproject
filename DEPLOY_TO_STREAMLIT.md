# Deploy JobGuard and get a public link

Streamlit Community Cloud deploys your app from a GitHub repository. After deployment, the app receives a public address ending in `.streamlit.app`.

## Before you begin

- Test the app locally with `python -m streamlit run app.py`.
- Create a free GitHub account if you do not already have one: <https://github.com/signup>.
- Keep `app.py`, `requirements.txt`, `.streamlit/config.toml`, and this document in the project folder. They are already present.

Do not upload `.venv`. It is local and is excluded by `.gitignore`.

## Step 1: Make a GitHub repository

1. Sign in to GitHub and select **New repository**.
2. Name it `fake-job-posting-detection`.
3. Choose **Public** if you want everyone to be able to view the code. Choose **Private** if you want to limit access; Streamlit Community Cloud can connect to either type of repository.
4. Do not add a README, `.gitignore`, or license during creation because the project already has a README and `.gitignore`.
5. Click **Create repository**.

## Step 2: Upload the project

The easiest method is GitHub's browser upload:

1. In the empty repository, click **uploading an existing file**.
2. Open `C:\Users\DIYA\OneDrive\Desktop\aiml project` in File Explorer.
3. Drag in `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `DEPLOY_TO_STREAMLIT.md`, `PROJECT_STUDY_GUIDE.md`, plus the folders `.streamlit` and `data`.
4. Do not drag in `.venv` or the original 47.7 MB dataset CSV. This app asks users to upload a CSV when it opens, so the dataset is not needed in the GitHub repository.
5. Add a commit message such as `Initial Streamlit app`, then click **Commit changes**.

If you prefer CMD and have Git installed, run these commands from the project folder after creating the repository. Replace `YOUR-USERNAME` with your GitHub username:

```cmd
cd /d "C:\Users\DIYA\OneDrive\Desktop\aiml project"
git init
git add .
git commit -m "Initial Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fake-job-posting-detection.git
git push -u origin main
```

GitHub may ask you to sign in or create a personal access token. The browser-upload method avoids that initial Git setup.

## Step 3: Deploy it

1. Visit <https://share.streamlit.io/> and sign in with the GitHub account that owns the repository.
2. Select **Create app**, then **Yup, I have an app**.
3. Select your repository: `YOUR-USERNAME/fake-job-posting-detection`.
4. Select branch: `main`.
5. Enter the main file path: `app.py`.
6. In **App URL**, enter an optional memorable name, for example `your-name-jobguard`. If available, the public URL becomes `https://your-name-jobguard.streamlit.app`.
7. Click **Deploy** and wait for the build to finish.

Streamlit installs the packages from `requirements.txt` automatically. It normally takes a few minutes for the first build. The build logs show the specific cause if deployment fails.

## After deployment

Open the public link and upload `fake_job_postings.csv`. Use `fraudulent` as the label and select `title`, `company_profile`, `description`, `requirements`, and `benefits` as text fields.

To update the live app later, edit the files in GitHub and commit the changes. Streamlit Community Cloud redeploys after it sees the changes. If you change dependencies, update and commit `requirements.txt` too.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` in the deployment logs | Confirm the library is listed in `requirements.txt` with the correct spelling. |
| App cannot find a file | Use paths relative to `app.py`. This app does not need a local dataset file because it uses the uploader. |
| Upload is rejected | The app is configured to allow uploads up to 200 MB. Your 47.7 MB CSV is within that limit. |
| Page shows old code | Confirm your GitHub commit contains the updated `app.py`, then use the app menu to reboot it if needed. |
| Public link needs a different name | Open the deployed app settings and change the app URL if the name is available. |

Official Streamlit deployment guide: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
