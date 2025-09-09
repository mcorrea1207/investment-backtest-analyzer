# 🚀 GitHub Pages Setup Guide

## Step 1: Initialize Git Repository (if not already done)

Run these commands in your terminal:

```bash
cd "/Users/pepe/Library/Mobile Documents/com~apple~CloudDocs/04. Macro/backtest_inversiones"
git init
git add index.html about.html README-website.md
git commit -m "Initial website setup"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com
2. Sign in (or create account if needed)
3. Click the green "New" button or go to https://github.com/new
4. Fill out:
   - **Repository name**: `investment-backtest-analyzer` (or any name you prefer)
   - **Description**: "Professional investment strategy backtesting and analysis tool"
   - **Visibility**: Public ✅ (required for free GitHub Pages)
   - **Initialize**: Leave unchecked (we already have files)

## Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands like this:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY-NAME` with your actual values.

## Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on "Settings" tab (top right of repository page)
3. Scroll down to "Pages" section in the left sidebar
4. Under "Source":
   - Select "Deploy from a branch"
   - Choose "main" branch
   - Choose "/ (root)" folder
   - Click "Save"

## Step 5: Access Your Website

Your website will be available at:
`https://YOUR-USERNAME.github.io/YOUR-REPOSITORY-NAME`

Note: It may take a few minutes for the site to become available.

## Step 6: Update Links (Optional)

Once your repository is public, update the GitHub links in your HTML files:

- In `index.html`, replace `href="#"` with your actual repository URL
- Example: `href="https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME"`

## 🎉 You're Done!

Your website will be live and free forever. Every time you push changes to the main branch, GitHub Pages will automatically update your website.

## Next Steps

- Share your website URL with others
- Add your actual repository links to the website
- Upload performance charts to make it more visual
- Consider adding Google Analytics for visitor tracking

## Troubleshooting

- **Site not loading?** Check the Pages settings and make sure the source is set correctly
- **Changes not showing?** It can take up to 10 minutes for changes to appear
- **404 error?** Make sure your files are in the root directory and named correctly

Need help? Check GitHub's documentation: https://docs.github.com/en/pages
