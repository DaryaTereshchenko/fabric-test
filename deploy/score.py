"""
Scoring script for the Superstore Forecast SARIMAX model.
This runs inside the Azure ML managed online endpoint container.
"""
import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    """Called once when the endpoint container starts. Loads the MLflow model."""
    global model
    model_path = os.getenv("AZUREML_MODEL_DIR")
    # Walk the model dir to find the MLmodel file (may be nested)
    for root, dirs, files in os.walk(model_path):
        if "MLmodel" in files:
            model_path = root
            break
    model = mlflow.statsmodels.load_model(model_path)
    logger.info("Model loaded from %s", model_path)


def run(raw_data: str) -> str:
    """
    Called on each scoring request.

    Expected JSON input:
    {
        "forecast_months": 6,         // number of months to forecast
        "start_date": "2023-08-01"    // optional: override forecast start date
    }

    Returns JSON with forecasted sales per month.
    """
    try:
        data = json.loads(raw_data)
        forecast_months = data.get("forecast_months", 6)
        start_date = data.get("start_date", None)

        if start_date:
            start = pd.Timestamp(start_date)
        else:
            # Default: forecast from the end of the training data
            start = model.fittedvalues.index[-1] + pd.DateOffset(months=1)

        end = start + pd.DateOffset(months=forecast_months - 1)

        predictions = model.get_prediction(start=start, end=end, dynamic=False)
        forecast_df = pd.DataFrame(
            {
                "Date": predictions.predicted_mean.index.strftime("%Y-%m-%d").tolist(),
                "Forecasted_Sales": predictions.predicted_mean.values.tolist(),
            }
        )

        return json.dumps(
            {
                "status": "success",
                "forecast_months": forecast_months,
                "predictions": forecast_df.to_dict(orient="records"),
            }
        )
    except Exception as e:
        logger.error("Scoring error: %s", str(e))
        return json.dumps({"status": "error", "message": str(e)})
