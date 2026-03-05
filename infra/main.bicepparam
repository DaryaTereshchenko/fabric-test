using 'main.bicep'

param mlWorkspaceName = '<YOUR_EXISTING_ML_WORKSPACE_NAME>'
param location = 'eastus'
param endpointName = 'superstore-forecast-endpoint'
param tags = {
  project: 'superstore-forecast'
  environment: 'dev'
}
