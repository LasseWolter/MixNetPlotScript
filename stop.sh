#!/bin/bash

# The quotes and the bracket let grep treat the string as a regex
# which means the grep command itself won't be matched since in the
# grep command the m is followed by an ']' not by an 'i'
ps ax | grep "[m]ix_net" | awk '{print "kill -9 " $1}' | bash
