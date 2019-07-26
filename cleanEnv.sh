#!/bin/bash
echo "Clean up environment by deleting old logs, statefiles 
and queue-outputs as well as already existing dbs"
sudo rm log/*.log 

sudo rm log/service_logs/*.log
sudo rm */*.db
