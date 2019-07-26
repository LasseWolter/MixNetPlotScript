#!/bin/bash
sed -i "s|QueueLogDir[= 0-9a-zA-Z_-\"]*|QueueLogDir=\"$1\"|g" provider/provider.toml service_provider/service_provider.toml mix1/mix1.toml
echo "Updated QueueLogDir to \"$1\", in all providers and mixes"


