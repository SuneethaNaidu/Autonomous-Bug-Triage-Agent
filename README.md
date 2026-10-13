# Autonomous Bug Triage Agent

A Streamlit mini project demonstrating an agentic-style pipeline for software bug triage: issue classification, potential duplicate detection, developer recommendation, explanations, review workflow, and analytics.

## Features
- Manual bug submission
- CSV batch triage
- Rule-based category/severity/priority baseline
- TF-IDF and cosine-similarity duplicate suggestions
- Expertise/workload-based developer recommendations
- Optional Gemini-generated explanations
- SQLite issue history
- Human review and approval
- Interactive Plotly dashboard and CSV export

> This is a research/demo prototype. It does not connect to Jira, GitHub Issues, or another external tracker. It does not automatically close, merge, or assign issues in an external system. Predictions are recommendations and should be reviewed.

## Run locally in VS Code

1. Install Python 3.11 or newer.
2. Extract the ZIP and open the `bug-triage-agent` folder in VS Code.
3. Open a terminal in that folder.
4. Create a virtual environment:

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL shown in the terminal, usually `http://localhost:8501`.

No API key is required for the main workflow.

## Optional Gemini explanations

1. Create an API key through Google's Gemini API tooling if available to you.
2. For local use, copy `.env.example` to `.env` and set `GEMINI_API_KEY=your_key`.
3. Restart Streamlit. Never commit `.env` or paste a key into public source code.
4. If Gemini is unavailable or its request fails, the app uses a local explanation fallback.

## Try the sample dataset

Go to **CSV Batch Triage** and upload `data/sample_issues.csv`. The CSV needs `title` and `description`; other supported columns are optional. Existing issue IDs are skipped to avoid overwriting records.

## Push to GitHub

Create an empty repository on GitHub. In the project folder:

```bash
git init
git add .
git commit -m "Initial autonomous bug triage agent"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example remote with your own repository URL. Do not upload `.env`, API keys, or private issue data.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch (`main`), and set the main file path to `app.py`.
4. If using Gemini, add `GEMINI_API_KEY` in the app's **Settings / Secrets** area. It is optional.
5. Deploy and test the public app.

The `requirements.txt` file must be at the repository root.

### Important hosting note
The SQLite database is suitable for local development and demos. On hosted services, local files may be ephemeral and can be reset when an app restarts or redeploys. For durable multi-user production storage, use a managed database such as PostgreSQL or another persistent database service. Do not store confidential company issue reports in a public demo.

## Limitations and next steps
- The classifier uses transparent keyword rules, not a trained classifier.
- Similarity is a heuristic and should be evaluated against labeled duplicate/non-duplicate pairs.
- The sample dataset is synthetic; its labels are illustrative, not verified ground truth.
- Team members and workloads are demo values.
- Future extensions: import/export from GitHub Issues or Jira, train and evaluate a supervised model, add audit logs, and use human-approved external tracker updates.

## Suggested evaluation
With a labeled test dataset, report classification precision/recall/F1, duplicate precision/recall at a chosen threshold, reviewer acceptance rate, and average triage time. Do not claim model accuracy until measured on held-out labeled data.
