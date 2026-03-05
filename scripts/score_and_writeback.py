"""
CI/CD script: Call the deployed Azure ML endpoint, then write predictions
back to the Fabric Lakehouse as a Delta table.

Usage:
  python scripts/score_and_writeback.py \
    --resource-group myRG \
    --ml-workspace myMLWorkspace \
    --endpoint-name superstore-forecast-endpoint
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from deltalake import write_deltalake


def get_credential() -> ClientSecretCredential:
    return ClientSecretCredential(
        tenant_id=os.environ["TENANT_ID"],
        client_id=os.environ["CLIENT_ID"],
        client_secret=os.environ["CLIENT_SECRET"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--ml-workspace", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--forecast-months", type=int, default=6)
    args = parser.parse_args()

    credential = get_credential()

    # Connect to Azure ML
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
        resource_group_name=args.resource_group,
        workspace_name=args.ml_workspace,
    )

    # Invoke the endpoint
    print(f"Invoking endpoint '{args.endpoint_name}'...")
    request_payload = json.dumps({"forecast_months": args.forecast_months})
    response = ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        request_file=None,
        deployment_name=None,
    )
    # If invoke doesn't support raw payload, use requests directly
    # For Azure ML SDK v2, use the invoke method or REST
    result = json.loads(response)
    print(f"  Status: {result['status']}")

    if result["status"] != "success":
        raise RuntimeError(f"Endpoint error: {result.get('message', 'unknown')}")

    # Build DataFrame from predictions
    predictions_df = pd.DataFrame(result["predictions"])
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
    predictions_df["Actual_Sales"] = np.nan
    predictions_df["MAPE"] = np.nan
    predictions_df["Category"] = "Furniture"

    # Write back to Fabric Lakehouse
    workspace_id = os.environ["WORKSPACE_ID"]
    lakehouse_id = os.environ["LAKEHOUSE_ID"]
    storage_scope = "https://storage.azure.com/.default"

    token = credential.get_token(storage_scope)
    storage_options = {
        "bearer_token": token.token,
        "use_fabric_endpoint": "true",
        "account_name": "onelake",
    }

    table_name = "Demand_Forecast_CICD"
    output_uri = f"az://{workspace_id}/{lakehouse_id}/Tables/{table_name}"

    write_deltalake(
        output_uri,
        predictions_df,
        mode="overwrite",
        storage_options=storage_options,
    )
    print(f"  Predictions written to Lakehouse table: {table_name}")


if __name__ == "__main__":
    main()
