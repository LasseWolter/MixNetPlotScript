import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime

class queueLog:
    def __init__(self,time,length):
        self.time = time
        self.length = length


timeStampFormat='%b %d %H:%M:%S.%f'
vals = {}
zips = {}
nodes=["ql_mix1","ql_provider","ql_serviceprovider"]

for node in nodes:
    vals[node]=[]
    with open(node) as f:
        data = f.readlines()
        for line in data:
            words = line.rstrip().split(",")
            time = datetime.datetime.strptime(words[0], timeStampFormat)
            entry = queueLog(time,int(words[1]))
            vals[node].append(entry)

for v in vals["ql_mix1"]:
    print(v.time, v.length)

# Find experiment start by finding earliest time stamp
times = []
for v in vals.values():
   times.append(v[0].time)

times.sort()
startTime = times[0]

times = {}
lengths = {}
for node,v in vals.items():
    times[node]=[]
    lengths[node]=[]
    for entry in v:
        entry.time = (entry.time - startTime).seconds
        times[node].append(entry.time)
        lengths[node].append(entry.length)


for node in nodes:
    plt.plot(times[node], lengths[node])

#plt.scatter(x, Y)

plt.show()

