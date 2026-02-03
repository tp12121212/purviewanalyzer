#!/usr/bin/env bash
set -euo pipefail

SUBSCRIPTION_ID="6fd92ebe-3092-45b6-83dd-20aeb921b9d0"
TENANT_ID="157925f3-de22-45dc-bd08-0138eb0625c9"
LOCATION="australiaeast"
RESOURCE_GROUP="kc-purview-alalyser-rg"
GITHUB_ORG="tp12121212"
GITHUB_REPO="purviewanalyzer"
GITHUB_BRANCH="main"
ACR_NAME="kcpurviewacr"
CONTAINER_APP_ENV="kc-purview-analyser-env"
CONTAINER_APP_NAME="kc-purview-analyser-app"
APP_DOMAIN="purview.killercloud.com.au"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subscription-id) SUBSCRIPTION_ID="$2"; shift 2 ;;
    --tenant-id) TENANT_ID="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
    --github-org) GITHUB_ORG="$2"; shift 2 ;;
    --github-repo) GITHUB_REPO="$2"; shift 2 ;;
    --github-branch) GITHUB_BRANCH="$2"; shift 2 ;;
    --acr-name) ACR_NAME="$2"; shift 2 ;;
    --containerapp-env) CONTAINER_APP_ENV="$2"; shift 2 ;;
    --containerapp-name) CONTAINER_APP_NAME="$2"; shift 2 ;;
    --app-domain) APP_DOMAIN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
 done

 echo "Logging into Azure..."
 az login --tenant "$TENANT_ID" >/dev/null
 az account set --subscription "$SUBSCRIPTION_ID"

 echo "Creating resource group (if missing)..."
 az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

 echo "Creating ACR (if missing)..."
 if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
   az acr create --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --sku Basic --location "$LOCATION" >/dev/null
 fi

 ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)
 ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)

 echo "Creating Container Apps environment (if missing)..."
 if ! az containerapp env show --name "$CONTAINER_APP_ENV" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
   az containerapp env create --name "$CONTAINER_APP_ENV" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null
 fi

 echo "Creating Container App (if missing)..."
 if ! az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
   BOOTSTRAP_IMAGE="mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
   az containerapp create \
     --name "$CONTAINER_APP_NAME" \
     --resource-group "$RESOURCE_GROUP" \
     --environment "$CONTAINER_APP_ENV" \
     --image "$BOOTSTRAP_IMAGE" \
     --target-port 8501 \
     --ingress external \
     --min-replicas 0 \
     --max-replicas 1 \
     --cpu 0.5 \
     --memory 1Gi >/dev/null
 fi

 APP_FQDN=$(az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

 echo "Creating App Registration for GitHub OIDC..."
 APP_NAME="$GITHUB_REPO-gha-oidc"
 APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv)
 if [[ -z "$APP_ID" ]]; then
   APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
 fi
 SP_ID=$(az ad sp list --filter "appId eq '$APP_ID'" --query "[0].id" -o tsv)
 if [[ -z "$SP_ID" ]]; then
   SP_ID=$(az ad sp create --id "$APP_ID" --query id -o tsv)
 fi

 RG_ID=$(az group show --name "$RESOURCE_GROUP" --query id -o tsv)
 echo "Assigning Contributor on resource group..."
 az role assignment create --assignee "$SP_ID" --role Contributor --scope "$RG_ID" >/dev/null
 echo "Assigning AcrPush on ACR..."
 az role assignment create --assignee "$SP_ID" --role AcrPush --scope "$ACR_ID" >/dev/null

 echo "Configuring federated credential for GitHub Actions..."
 FEDERATED_NAME="github-actions-$GITHUB_REPO"
 SUBJECT="repo:$GITHUB_ORG/$GITHUB_REPO:ref:refs/heads/$GITHUB_BRANCH"
 if ! az ad app federated-credential list --id "$APP_ID" --query "[?name=='$FEDERATED_NAME']" -o tsv | grep -q .; then
   az ad app federated-credential create --id "$APP_ID" --parameters "{\"name\":\"$FEDERATED_NAME\",\"issuer\":\"https://token.actions.githubusercontent.com\",\"subject\":\"$SUBJECT\",\"description\":\"GitHub Actions OIDC\",\"audiences\":[\"api://AzureADTokenExchange\"]}" >/dev/null
 fi

 echo "\n=== GitHub Secrets to set ==="
 echo "AZURE_CLIENT_ID=$APP_ID"
 echo "AZURE_TENANT_ID=$TENANT_ID"
 echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
 echo "AZURE_RESOURCE_GROUP=$RESOURCE_GROUP"
 echo "AZURE_CONTAINERAPP_NAME=$CONTAINER_APP_NAME"
 echo "AZURE_CONTAINERAPP_ENV=$CONTAINER_APP_ENV"
 echo "AZURE_ACR_NAME=$ACR_NAME"
 echo "AZURE_ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER"
 echo "\n=== DNS (Container App custom domain) ==="
 echo "Create CNAME: $APP_DOMAIN -> $APP_FQDN"
 echo "After DNS propagates, bind the domain with a certificate:"
 echo "az containerapp hostname add --resource-group \"$RESOURCE_GROUP\" --name \"$CONTAINER_APP_NAME\" --hostname \"$APP_DOMAIN\" --certificate <CERT_NAME>"
 echo "\n=== DNS (optional for ACR custom domain; Premium only) ==="
 echo "Create CNAME: acr.killercloud.com.au -> $ACR_LOGIN_SERVER"
