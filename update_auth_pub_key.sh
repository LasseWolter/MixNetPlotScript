#!/bin/bash
echo "Updating all configs of mixes and providers to the new auth pub key: $1"
sed -i "s|PublicKey[ \"=a-zA-Z0-9]*|PublicKey = \"$1\"|g" mix1/mix1.toml provider/provider.toml service_provider/service_provider.toml 
