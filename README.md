# Superstore Sales Forecast — MLOps with Azure ML & Microsoft Fabric

End-to-end MLOps pipeline that trains a SARIMAX time-series model on Superstore sales data stored in a **Microsoft Fabric Lakehouse**, deploys it as a real-time endpoint in **Azure Machine Learning**, and writes predictions back to Fabric.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Fabric         │      │  GitHub Actions   │      │  Azure ML           │
│  Lakehouse      │◄────►│  CI/CD Pipeline   │─────►│  Managed Endpoint   │
│  (OneLake)      │      │                   │      │  (SARIMAX model)    │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
```

**Pipeline stages:**

1. **Infra** — Deploys the managed online endpoint into an existing Azure ML workspace via Bicep
2. **Train** — Reads data from Fabric Lakehouse, trains SARIMAX, registers the model in Azure ML (and optionally in Fabric's MLflow registry)
3. **Deploy** — Creates a versioned deployment with blue-green rollout, runs a smoke test
4. **Write-back** — Invokes the endpoint and writes forecast predictions back to Fabric Lakehouse as a Delta table

## Repository Structure

```
├── .github/workflows/
│   └── mlops-pipeline.yml          # GitHub Actions CI/CD pipeline
├── deploy/
│   ├── conda.yml                   # Conda environment for the endpoint container
│   ├── deployment.yml              # Azure ML managed online deployment spec
│   ├── endpoint.yml                # Azure ML managed online endpoint spec
│   ├── sample-request.json         # Sample scoring request for smoke tests
│   └── score.py                    # Scoring script (init/run) for the endpoint
├── infra/
│   ├── main.bicep                  # Bicep template — deploys endpoint into existing ML workspace
│   └── main.bicepparam             # Bicep parameters
├── scripts/
│   ├── train_and_register.py       # Train SARIMAX, log to MLflow, register model
│   └── score_and_writeback.py      # Call endpoint, write predictions to Lakehouse
├── AIsample - Superstore Forecast.ipynb  # Interactive notebook for exploration
└── requirements-ci.txt             # Python dependencies for CI/CD scripts
```

## Prerequisites

- **Azure ML workspace** (existing) with a linked storage account
- **Microsoft Fabric workspace** with a Lakehouse containing the Superstore dataset at `Files/salesforecast/raw/Superstore.xlsx`
- **Fabric capacity** in an active (not paused) state
- **Entra ID service principal** with:
  - `Contributor` role on the Azure resource group
  - `Storage Blob Data Contributor` role on the ML workspace storage account
  - `Contributor` access on the Fabric workspace (for Lakehouse read/write and optional MLflow registration)

## GitHub Secrets

Configure these in your repository settings (**Settings → Secrets and variables → Actions**):

| Secret | Description |
|--------|-------------|
| `CLIENT_ID` | Entra app (service principal) client ID |
| `CLIENT_SECRET` | Entra app client secret |
| `TENANT_ID` | Entra ID tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | Resource group containing the Azure ML workspace |
| `AZURE_ML_WORKSPACE_NAME` | Name of the existing Azure ML workspace |
| `WORKSPACE_ID` | Microsoft Fabric workspace ID |
| `LAKEHOUSE_ID` | Microsoft Fabric Lakehouse ID |

## Running the Pipeline

The pipeline triggers automatically on pushes to `main` that modify relevant files. You can also trigger it manually via **Actions → MLOps – Train, Deploy & Write-back to Fabric → Run workflow**.

## Model Details

- **Algorithm:** SARIMAX (Seasonal ARIMA with eXogenous variables)
- **Order:** (0, 1, 1) × (0, 1, 1, 12)
- **Category:** Furniture sales forecasting
- **Metric:** MAPE (Mean Absolute Percentage Error)
- **Framework:** statsmodels, logged via MLflow

## Endpoint Usage

Once deployed, invoke the endpoint:

```json
POST https://<scoring-uri>
Authorization: Bearer <endpoint-key>
Content-Type: application/json

{
    "forecast_months": 6,
    "start_date": "2024-01-01"
}
```

Response:

```json
{
    "status": "success",
    "forecast_months": 6,
    "predictions": [
        {"Date": "2024-01-01", "Forecasted_Sales": 12345.67},
        ...
    ]
}
```

## Viewing Results

| What | Where |
|------|-------|
| Registered model | [Azure ML Studio](https://ml.azure.com) → Models → `superstore-sarimax` |
| Deployed endpoint | Azure ML Studio → Endpoints → `superstore-forecast-endpoint` |
| MLflow experiment | Azure ML Studio → Jobs → `superstore-forecast-cicd` |
| Predictions table | Fabric → Lakehouse → Tables → `Demand_Forecast_CICD` |
