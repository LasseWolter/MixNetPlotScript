#!/bin/bash
if [ -z "$1" ];
then
    echo "Usage ./runExperiment.sh <expConfig>"
    exit 
fi

expConf=$1
curExpDir="./exps/curExp/"
root="/home/lasse/Programming/go_projects/src/github.com/katzenpost/"

printf "\n >>>You are running an experiment with the config: $expConf<<<\n\n"

# Clean up the environment by deleting already existing dbs, statefiles, etc.
source ./cleanEnv.sh
echo "Installing go binaries..."
go install ${root}daemons/server
go install ${root}daemons/authority/nonvoting
go install ${root}catshadow/cmd
go install ${root}catshadow/experiment
echo "...finished installing binaries."
echo "------------------------------"

# Copy exp config into curExp dir
\rm -rf ${curExpDir}*
cp $expConf ${curExpDir}alice.toml

printf "Copying most recent go binaries for
\n\t-server\n\t-nonvoting\n\t-panda\n\t-spool_server\n\t-cmd\n\t-experiment\n\n"
                                            
cp /home/lasse/Programming/go_projects/bin/{server,nonvoting,panda,spool_server,experiment} ./bin

# Update alice.toml
#sed -i "s|Duration[= 0-9]*\>|Duration=$expDuration|g" bin/alice.toml 
#sed -i "s|LambdaP[= 0-9.]*\>|LambdaP=$LambdaP|g" bin/alice.toml
#sed -i "s|LambdaPMaxDelay[= 0-9.]*\>|LambdaPMaxDelay=$LambdaPMaxDelay|g" bin/alice.toml


#source ./update_QueuePollInterval.sh $QueuePollInterval
#
#echo "Updating config files..."
#source ./update_QueueLogDir.sh $QueueLogDir 


printf "\nStart buiding docker container...\n"
sudo docker-compose build
printf "Finished building docker container.\n\n"

printf "Starting docker container in using docker-compose up...\n"
docker-compose up &
