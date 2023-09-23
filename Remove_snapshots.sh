#!/bin/bash

set -e

# **************************************************************
# This script removes bad/unused snapshots in a range of indices
# **************************************************************


######################### USER-DEFINED VARIABLES ###############################

path="/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08/smallb2_xy"
basename="smallb2_xy_"
extension=".png"

idx1=1405
idx2=1458

echo_only=false

################################################################################


for ((i=idx1; i<=idx2; ++i))
do
    fullpath=$path
    fullpath+="/"
    fullpath+=$basename
    formatted_i="$(printf "%04d" $i)"
    fullpath+=$formatted_i
    fullpath+=$extension

    if $echo_only
    then
        echo "rm $fullpath"
    else
        rm $fullpath
    fi
done
