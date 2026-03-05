// MLOps Infrastructure for Superstore Forecast Model
// Deploys only the Managed Online Endpoint into an EXISTING Azure ML Workspace.
// The workspace (and its storage account, key vault, ACR, App Insights)
// are assumed to already exist. No duplicate resources are created.
//
// Usage: az deployment group create -g <rg> -f main.bicep -p main.bicepparam

targetScope = 'resourceGroup'

@description('Name of your existing Azure ML workspace')
param mlWorkspaceName string

@description('Azure region (must match the existing workspace)')
param location string = resourceGroup().location

@description('Name of the managed online endpoint to create')
param endpointName string = 'superstore-forecast-endpoint'

@description('Tags applied to the endpoint')
param tags object = {
  project: 'superstore-forecast'
  environment: 'dev'
}

// Reference the EXISTING Azure ML Workspace (no new resources created)
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = {
  name: mlWorkspaceName
}

// Managed Online Endpoint for real-time inference (the only new resource)
resource onlineEndpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-04-01' = {
  parent: mlWorkspace
  name: endpointName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    authMode: 'Key'
  }
}

// Outputs
output mlWorkspaceName string = mlWorkspace.name
output mlWorkspaceId string = mlWorkspace.id
output endpointName string = onlineEndpoint.name
output scoringUri string = onlineEndpoint.properties.scoringUri
