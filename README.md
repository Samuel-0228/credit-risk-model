# Credit Risk Model for Buy-Now-Pay-Later Service

## Overview
This repository implements an end-to-end credit risk probability model using alternative data from an eCommerce platform. The model leverages RFM (Recency, Frequency, Monetary) behavioral patterns to create a proxy for credit risk, enabling loan approvals for buy-now-pay-later services at Bati Bank.

Key components:
- Feature engineering with WoE/IV transformations.
- Multiple ML models (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning.
- MLflow for experiment tracking.
- FastAPI deployment with Docker and CI/CD via GitHub Actions.


## Installation
1. Clone the repo: `git clone https://github.com/Samuel-0228/credit-risk-model.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download data from [Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge) and place in `data/raw/`.

## Usage
- Run EDA: `jupyter notebook notebooks/eda.ipynb`
- Train models: `python src/train.py`
- Start API: `uvicorn src.api.main:app --reload`

## Documentation

### Credit Scoring Business Understanding
This section summarizes key concepts from credit risk fundamentals and regulatory frameworks, tailored to our project.

#### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Accord, introduced in 2004 by the Basel Committee on Banking Supervision, builds on Basel I by emphasizing three pillars: minimum capital requirements based on risk-weighted assets, supervisory review, and market discipline through disclosures.<grok-card data-id="3779d0" data-type="citation_card"></grok-card> Its focus on advanced risk measurement—requiring banks to use internal ratings-based (IRB) approaches for estimating probability of default (PD), loss given default (LGD), and exposure at default (EAD)—demands models that accurately quantify credit risk to ensure capital adequacy (at least 8% of risk-weighted assets).<grok-card data-id="351448" data-type="citation_card"></grok-card><grok-card data-id="482eec" data-type="citation_card"></grok-card> This influences our project by necessitating interpretable models (e.g., Logistic Regression with WoE) that allow regulators to validate assumptions, trace feature impacts on predictions, and ensure compliance during supervisory reviews. Well-documented models, including validation reports and back-testing (e.g., ROC-AUC stability), mitigate model risk under Pillar 2, preventing undercapitalization from opaque predictions and enabling transparent disclosures under Pillar 3.<grok-card data-id="0b3937" data-type="citation_card"></grok-card>

#### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
In traditional credit scoring, a direct "default" label (e.g., 90+ days past due per Basel II) is derived from historical loan performance.<grok-card data-id="06563e" data-type="citation_card"></grok-card> Our eCommerce transaction data lacks this, so a proxy (e.g., RFM-based clustering identifying "disengaged" customers as high-risk) is essential to simulate default likelihood using behavioral signals like low transaction frequency or recency, transforming alternative data into a predictive risk signal.<grok-card data-id="c560b0" data-type="citation_card"></grok-card> This enables model training for loan approvals without historical defaults.

However, proxy-based predictions carry business risks: (1) **Bias amplification**—proxies like transaction patterns may inadvertently proxy protected characteristics (e.g., income via spending), leading to discriminatory outcomes and regulatory violations (e.g., ECOA);<grok-card data-id="05bff8" data-type="citation_card"></grok-card> (2) **Inaccurate risk signals**—if the proxy poorly correlates with true defaults, it could approve high-risk loans (increasing losses) or reject low-risk ones (lost revenue); (3) **Privacy and compliance issues**—behavioral data raises GDPR/PDPO concerns, eroding trust if mishandled.<grok-card data-id="25e724" data-type="citation_card"></grok-card> Mitigation includes fairness audits and continuous validation against real defaults post-launch.

#### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple models like Logistic Regression with Weight of Evidence (WoE) transformations offer high interpretability—coefficients directly show feature impacts on log-odds of default, aligning with regulatory needs for explainability (e.g., tracing decisions in consumer disputes).<grok-card data-id="701132" data-type="citation_card"></grok-card> They are computationally efficient, easier to validate, and compliant with Basel II's IRB requirements, but may underperform on non-linear patterns in alternative data, leading to lower AUC (e.g., 0.75 vs. 0.85 for complex models).

Complex models like Gradient Boosting (e.g., XGBoost) excel in performance, handling interactions and non-linearity for better PD predictions, but are "black-box," complicating regulatory scrutiny and bias detection.<grok-card data-id="0939d5" data-type="citation_card"></grok-card> In regulated contexts, trade-offs include: (1) **Compliance vs. Accuracy**—simple models speed approvals but risk suboptimal capital allocation; complex ones boost inclusion but require add-ons like LIME/SHAP for interpretability, increasing costs; (2) **Deployment Risks**—complex models amplify overfitting in sparse data, per SR 11-7 guidelines;<grok-card data-id="c66cae" data-type="citation_card"></grok-card> (3) **Fairness**—both need bias checks, but simple models facilitate easier audits. We prioritize a hybrid: start with Logistic for interpretability, benchmark against boosting for performance gains.

## Next Steps
- Feature engineering and proxy target creation (Task 3–4).
- Model training with MLflow (Task 5).

## References
- Basel II Framework: [BIS](https://www.bis.org/publ/bcbsca.htm)
- Alternative Credit Scoring: [HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- World Bank Guidelines: [World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)

## License
MIT