param(
  [string]$SubscriptionId = "6fd92ebe-3092-45b6-83dd-20aeb921b9d0",
  [string]$TenantId = "157925f3-de22-45dc-bd08-0138eb0625c9",
  [string]$Location = "australiaeast",
  [string]$ResourceGroup = "kc-purview-alalyser-rg",
  [string]$GitHubOrg = "tp12121212",
  [string]$GitHubRepo = "purviewanalyzer",
  [string]$GitHubBranch = "main",
  [string]$AcrName = "kcpurviewacr",
  [string]$ContainerAppEnv = "kc-purview-analyser-env",
  [string]$ContainerAppName = "kc-purview-analyser-app",
  [string]$AppDomain = "purview.killercloud.com.au"
)

$ErrorActionPreference = "Stop"

Write-Host "Logging into Azure..."
az login --tenant $TenantId | Out-Null
az account set --subscription $SubscriptionId

Write-Host "Creating resource group (if missing)..."
az group create --name $ResourceGroup --location $Location | Out-Null

Write-Host "Creating ACR (if missing)..."
$acr = az acr show --name $AcrName --resource-group $ResourceGroup 2>$null
if (-not $acr) {
  az acr create --name $AcrName --resource-group $ResourceGroup --sku Basic --location $Location | Out-Null
}

$acrLoginServer = az acr show --name $AcrName --resource-group $ResourceGroup --query loginServer -o tsv
$acrId = az acr show --name $AcrName --resource-group $ResourceGroup --query id -o tsv

Write-Host "Creating Container Apps environment (if missing)..."
$env = az containerapp env show --name $ContainerAppEnv --resource-group $ResourceGroup 2>$null
if (-not $env) {
  az containerapp env create --name $ContainerAppEnv --resource-group $ResourceGroup --location $Location | Out-Null
}

Write-Host "Creating Container App (if missing)..."
$app = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup 2>$null
if (-not $app) {
  $bootstrapImage = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
  az containerapp create `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --environment $ContainerAppEnv `
    --image $bootstrapImage `
    --target-port 8501 `
    --ingress external `
    --min-replicas 0 `
    --max-replicas 1 `
    --cpu 0.5 `
    --memory 1Gi | Out-Null
}

Write-Host "Fetching Container App FQDN..."
$appFqdn = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv

Write-Host "Creating App Registration for GitHub OIDC..."
$appName = "$GitHubRepo-gha-oidc"
$appId = az ad app list --display-name $appName --query "[0].appId" -o tsv
if (-not $appId) {
  $appId = az ad app create --display-name $appName --query appId -o tsv
}
$spId = az ad sp list --filter "appId eq '$appId'" --query "[0].id" -o tsv
if (-not $spId) {
  $spId = az ad sp create --id $appId --query id -o tsv
}

$rgId = az group show --name $ResourceGroup --query id -o tsv
Write-Host "Assigning Contributor on resource group..."
az role assignment create --assignee $spId --role Contributor --scope $rgId | Out-Null
Write-Host "Assigning AcrPush on ACR..."
az role assignment create --assignee $spId --role AcrPush --scope $acrId | Out-Null

Write-Host "Configuring federated credential for GitHub Actions..."
$federatedName = "github-actions-$GitHubRepo"
$subject = "repo:$GitHubOrg/$GitHubRepo:ref:refs/heads/$GitHubBranch"
$federated = az ad app federated-credential list --id $appId --query "[?name=='$federatedName']" -o tsv
if (-not $federated) {
  $federatedBody = @{
    name = $federatedName
    issuer = "https://token.actions.githubusercontent.com"
    subject = $subject
    description = "GitHub Actions OIDC"
    audiences = @("api://AzureADTokenExchange")
  } | ConvertTo-Json -Depth 5
  az ad app federated-credential create --id $appId --parameters $federatedBody | Out-Null
}

Write-Host "\n=== GitHub Secrets to set ==="
Write-Host "AZURE_CLIENT_ID=$appId"
Write-Host "AZURE_TENANT_ID=$TenantId"
Write-Host "AZURE_SUBSCRIPTION_ID=$SubscriptionId"
Write-Host "AZURE_RESOURCE_GROUP=$ResourceGroup"
Write-Host "AZURE_CONTAINERAPP_NAME=$ContainerAppName"
Write-Host "AZURE_CONTAINERAPP_ENV=$ContainerAppEnv"
Write-Host "AZURE_ACR_NAME=$AcrName"
Write-Host "AZURE_ACR_LOGIN_SERVER=$acrLoginServer"
Write-Host "\n=== DNS (Container App custom domain) ==="
Write-Host "Create CNAME: $AppDomain -> $appFqdn"
Write-Host "After DNS propagates, bind the domain with a certificate:"
Write-Host "az containerapp hostname add --resource-group $ResourceGroup --name $ContainerAppName --hostname $AppDomain --certificate <CERT_NAME>"
Write-Host "\n=== DNS (optional for ACR custom domain; Premium only) ==="
Write-Host "Create CNAME: acr.killercloud.com.au -> $acrLoginServer"
