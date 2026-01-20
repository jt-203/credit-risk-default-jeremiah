# Credit Risk Default Modeling & Dashboard

Author: **Jeremiah Tshinyama**

This project analyzes credit risk for personal loans using historical loan-level data.
It trains machine learning models to predict the probability of default at origination
and provides an interactive Streamlit dashboard to explore risk factors and simulate
lending policies.

## Project Structure

- `src/train_models.py` — loads data, preprocesses features, trains models, saves artifacts
- `src/app.py` — Streamlit dashboard that loads trained models and lets users explore risk
- `requirements.txt` — Python dependencies
- `data/` — (you create this locally) put your LendingClub CSV here

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your LendingClub CSV in a local `data/` folder.
4. Run model training:

   ```bash
   python src/train_models.py --data_path data/your_lendingclub_file.csv
   ```

5. Launch the dashboard:

   ```bash
   streamlit run src/app.py
   ```

You can customize feature engineering, models, and dashboard layout as you iterate.
