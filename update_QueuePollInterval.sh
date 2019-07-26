#!/bin/bash
sed -i "s|QueuePollInterval[= 0-9]*|QueuePollInterval=$1|g" provider/provider.toml service_provider/service_provider.toml mix1/mix1.toml
echo "Updated QueuePollInterval to $1 ms, in all providers and mixes"


