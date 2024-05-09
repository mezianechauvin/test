#!/bin/sh
> .env
echo "CURRENT_USERNAME=eddie.deane@greenchef.com" >> .env
echo "VAULT_TOKEN=$(vault login -address=https://vault.secrets.hellofresh.io -method=oidc -namespace=infrastructure/data-science-spew -token-only)" >> .env
