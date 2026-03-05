"""
CI/CD script: Train the SARIMAX model on data from the Fabric Lakehouse
and register it in Azure ML.

Usage:
  python scripts/train_and_register.py \
    --resource-group myRG \
    --ml-workspace myMLWorkspace \
    --model-name superstore-sarimax
"""
import argparse
import io
import os

import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
from deltalake import DeltaTable
from sklearn.metrics import mean_absolute_percentage_error


def get_credential() -> ClientSecretCredential:
    return ClientSecretCredential(
        tenant_id=os.environ["TENANT_ID"],
        client_id=os.environ["CLIENT_ID"],
        client_secret=os.environ["CLIENT_SECRET"],
    )


def load_data_from_lakehouse(credential: ClientSecretCredential) -> pd.DataFrame:
    """Load the Superstore dataset from the Fabric Lakehouse via OneLake."""
    workspace_id = os.environ["WORKSPACE_ID"]
    lakehouse_id = os.environ["LAKEHOUSE_ID"]
    storage_scope = "https://storage.azure.com/.default"

    token = credential.get_token(storage_scope)
    storage_options = {
        "bearer_token": token.token,
        "use_fabric_endpoint": "true",
        "account_name": "onelake",
    }
    delta_table_path = f"{lakehouse_id}/Tables/Superstore"
    dt = DeltaTable(
        table_uri=f"az://{workspace_id}/{delta_table_path}",
        storage_options=storage_options,
    )
    return dt.to_pandas()


def preprocess(df: pd.DataFrame) -> pd.Series:
    """Filter to Furniture, aggregate monthly, and shift dates."""
    furniture = df.loc[df["Category"] == "Furniture"].copy()
    cols_to_drop = [
        "Row ID", "Order ID", "Ship Date", "Ship Mode", "Customer ID",
        "Customer Name", "Segment", "Country", "City", "State",
        "Postal Code", "Region", "Product ID", "Category",
        "Sub-Category", "Product Name", "Quantity", "Discount", "Profit",
    ]
    furniture.drop(columns=[c for c in cols_to_drop if c in furniture.columns], inplace=True)
    furniture = furniture.sort_values("Order Date")
    furniture = furniture.groupby("Order Date")["Sales"].sum().reset_index()
    furniture = furniture.set_index("Order Date")
    y = furniture["Sales"].resample("MS").mean()
    y = y.reset_index()
    y["Order Date"] = pd.to_datetime(y["Order Date"])
    y["Order Date"] = [i + pd.DateOffset(months=67) for i in y["Order Date"]]
    y = y.set_index(["Order Date"])
    return y["Sales"]


def train_model(y: pd.Series):
    """Train SARIMAX(0,1,1)(0,1,1,12) and return the fitted results."""
    mod = sm.tsa.statespace.SARIMAX(
        y,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return mod.fit(disp=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--ml-workspace", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    credential = get_credential()

    # Connect to Azure ML
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
        resource_group_name=args.resource_group,
        workspace_name=args.ml_workspace,
    )
    tracking_uri = ml_client.workspaces.get(args.ml_workspace).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("superstore-forecast-cicd")

    # Load data from Fabric Lakehouse
    print("Loading data from Fabric Lakehouse...")
    df = load_data_from_lakehouse(credential)
    print(f"  Loaded {len(df)} rows")

    # Preprocess
    y = preprocess(df)
    print(f"  Monthly time series: {len(y)} observations")

    # Train
    print("Training SARIMAX model...")
    with mlflow.start_run(run_name="cicd-train") as run:
        results = train_model(y)

        # Log metrics
        max_date = y.reset_index()["Order Date"].max()
        predictions = results.get_prediction(
            start=max_date - pd.DateOffset(months=5), dynamic=False
        )
        y_truth = y[max_date - pd.DateOffset(months=5):]
        mape = mean_absolute_percentage_error(y_truth.values, predictions.predicted_mean.values)
        mlflow.log_metric("mape", mape)
        mlflow.log_params({
            "order": "(0,1,1)",
            "seasonal_order": "(0,1,1,12)",
        })

        # Log and register the model
        mlflow.statsmodels.log_model(
            results,
            artifact_path=args.model_name,
            registered_model_name=args.model_name,
        )
        print(f"  Model registered: {args.model_name}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  MAPE: {mape:.4f}")

    mlflow.end_run()


if __name__ == "__main__":
    main()
