# Free deployment: dashboard + website with a free domain

You will get two free HTTPS addresses:

| What | Service | Address looks like |
|---|---|---|
| Interactive dashboard | Streamlit Community Cloud | `https://dan-cake-analytics.streamlit.app` |
| Project website (+ PDF report) | GitHub Pages | `https://sumon-cse83.github.io/dan-cake-analytics` |

Both are free forever for public projects. A paid custom domain (e.g. `dancake-analytics.com`, about USD 10/year)
can be attached later. See step 5.

## 1. Put the project on GitHub (once)

1. Create a free account at https://github.com and a **public** repository, e.g. `dan-cake-analytics`.
2. Install Git (already installed on this PC) and run in the project folder:

```powershell
git init
git add .
git commit -m "Dan Cake analytics framework"
git branch -M main
git remote add origin https://github.com/sumon-cse83/dan-cake-analytics.git
git push -u origin main
```

`.gitignore` keeps the 80 MB raw CSV and other large files out of GitHub. The dashboard only needs
`data/dashboard/` (about 2 MB), which **is** uploaded.

## 2. Dashboard → Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository `sumon-cse83/dan-cake-analytics`, branch `main`, main file path `dashboard/app.py`.
4. *Advanced settings → Python version*: 3.12 or newer.
5. Optionally choose a custom sub-domain, e.g. `dan-cake-analytics`, then click **Deploy**.

The app reads `dashboard/requirements.txt` (small: pandas, numpy, plotly, streamlit, pyarrow).

## 3. Website → GitHub Pages

1. Regenerate the site with your real links:

```powershell
cd src
python -m dancake.website --dashboard-url https://dan-cake-analytics.streamlit.app --repo-url https://github.com/sumon-cse83/dan-cake-analytics
cd ..
copy report\main.pdf docs\report.pdf
git add docs; git commit -m "website"; git push
```

2. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs` → Save**.
3. After 1–2 minutes the site is live at `https://sumon-cse83.github.io/dan-cake-analytics`.

## 4. Updating after new data

```powershell
python run_pipeline.py
cd src; python -m dancake.website --dashboard-url ... --repo-url ...; cd ..
git add -A; git commit -m "update data"; git push
```

Streamlit Cloud and GitHub Pages redeploy automatically on every push.

## 5. (Optional) Own domain name

Buy a domain from any registrar. In its DNS settings add a `CNAME` record `www → sumon-cse83.github.io`.
Then, in GitHub **Settings → Pages → Custom domain**, enter `www.your-domain.com` and tick **Enforce HTTPS**.
Free alternatives such as `*.github.io`, `*.streamlit.app`, `*.netlify.app` and `*.pages.dev` need no purchase.
