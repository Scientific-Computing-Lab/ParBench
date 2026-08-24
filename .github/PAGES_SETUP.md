# GitHub Pages Setup

## Setting the Pages Password

The visualizations site is password-protected via staticrypt. Before the workflow can deploy, set the password secret:

### Via GitHub UI
1. Go to repo Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PAGES_PASSWORD`
4. Value: your chosen password
5. Click "Add secret"

### Via GitHub API (if gh CLI is available)
```bash
gh secret set PAGES_PASSWORD
```

## Re-deploying
After setting the secret, trigger a new deployment:
- Push any change to `visualizations/`
- Or go to Actions → "Deploy Visualizations to GitHub Pages" → "Run workflow"

## Changing the Password
Update the `PAGES_PASSWORD` secret and re-run the workflow.
