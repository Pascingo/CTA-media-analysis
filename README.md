# CTA-media-analysis
Repository for our group work in Computational Language Technology.

Since we are working collaboratively in Google Colab and the Kaggle dataset is quite large, we use the Kaggle API to download the data directly into our active runtime. 

To ensure our API tokens remain secure and are not exposed in the codebase on GitHub, we use **Google Colab Secrets**. Every team member only needs to configure their credentials once in their own Colab environment.

## Prerequisites: Setup Instructions

Before running the notebooks, please complete the following steps:

### 1. Generate Your Kaggle API Token
1. Log in to your [Kaggle](https://www.kaggle.com/) account.
2. Go to your profile **Settings**.
3. Scroll down to the **API** section and click **Create New Token**.
4. You will receive a new token (usually starting with `KGAT_...`). Keep this token and your Kaggle username ready.

### 2. Configure Colab Secrets
1. Open the project notebook in Google Colab.
2. In the left-hand sidebar, click on the **Key icon** (Secrets).
3. Click **Add new secret** and create the first variable:
   * **Name:** `KAGGLE_USERNAME`
   * **Value:** `<your-kaggle-username>`
   * ⚠️ **Important:** Toggle the switch to grant **Notebook access**.
4. Add a second secret:
   * **Name:** `KAGGLE_KEY`
   * **Value:** `<your-KGAT-token>`
   * ⚠️ **Important:** Toggle the switch to grant **Notebook access**.

### 3. Run the Download Cell
Once your secrets are saved, you can simply run the first cell in the notebook. The script will securely load your credentials from the Colab vault, authenticate with Kaggle, and download the `ai-media-dataset` directly into your session without any manual uploads.
