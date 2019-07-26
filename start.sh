#!/bin/bash

num_mixes=1

binDir="/mix_net/bin/"
cliConf="/mix_net/exps/curExp/alice.toml"

./stop.sh

echo "------------------------------"
echo "Starting up Authority"
echo "------------------------------"
    ${binDir}nonvoting -f /mix_net/auth/authority.toml &
    sleep 0.5 


echo "------------------------------"
echo "Starting up Providers"
echo "------------------------------"
    ${binDir}server -f /mix_net/provider/provider.toml -c $cliConf &
    echo "Started provider"
    sleep 0.5 
    ${binDir}server -f service_provider/service_provider.toml -c $cliConf &
    echo "Started service_provider"
    sleep 0.5 


echo "------------------------------"
echo "Starting up $num_mixes Mixes"
echo "------------------------------"
for i in `seq 1 $num_mixes`
do
    ${binDir}server -f /mix_net/mix$i/mix$i.toml -c $cliConf &
    echo "Started up Mix $i"
    sleep 0.5 
done

# Wait until the first Handshake completes (+a few secs) - this means the provider is connencted to the mix
echo "Waiting for Handshake between servers to complete..."
grep -m 1 "Handshake completed" <(tail -f /mix_net/log/provider.log)
sleep 10 
/mix_net/bin/experiment -f $cliConf 

echo "Waiting until Queue length of last node is empty"
grep -m 1 "Current Queue length: 0" <(tail -f /mix_net/log/mix1.log)
# After the experiment finished, the docker container will close itself
