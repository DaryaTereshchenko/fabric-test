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
import tempfile

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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(request_payload)
        request_file = f.name
    response = ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        request_file=request_file,
    )
    os.unlink(request_file)
    # The endpoint returns a JSON string; invoke() returns it as-is.
    # It may need one or two json.loads() calls depending on encoding.
    result = json.loads(response)
    if isinstance(result, str):
        result = json.loads(result)
    print(f"  Status: {result['status']}")

    if result["status"] != "success":
        raise RuntimeError(f"Endpoint error: {result.get('message', 'unknown')}")

    # Build DataFrame from predictions
    predictions_df = pd.DataFrame(result["predictions"])
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
    predictions_df["Actual_Sales"] = np.nan
    predictions_df["MAPE"] = np.nan
    predictions_df["Category"] = "Furniture"

    # Write back to Fabric Lakehouse via DataLakeServiceClient
    workspace_id = os.environ["WORKSPACE_ID"]
    lakehouse_id = os.environ["LAKEHOUSE_ID"]

    from azure.storage.filedatalake import DataLakeServiceClient

    datalake_client = DataLakeServiceClient(
        account_url="https://onelake.dfs.fabric.microsoft.com",
        credential=credential,
    )
    fs_client = datalake_client.get_file_system_client(workspace_id)

    table_name = "Demand_Forecast_CICD"
    table_dir = f"{lakehouse_id}/Tables/{table_name}"

    # Write as parquet locally then upload
    local_parquet = "/tmp/forecast_output.parquet"
    predictions_df.to_parquet(local_parquet, index=False)

    # Upload the parquet file
    parquet_path = f"{table_dir}/part-00000.parquet"
    dir_client = fs_client.get_directory_client(table_dir)
    dir_client.create_directory()
    file_client = fs_client.get_file_client(parquet_path)
    with open(local_parquet, "rb") as f:
        data = f.read()
    file_client.upload_data(data, overwrite=True)
    os.unlink(local_parquet)
    print(f"  Predictions written to Lakehouse: {table_dir}")


if __name__ == "__main__":
    main()
