#!/bin/bash

set -e

# ******************************************************************************
# This script generates symlinks to user-specified paths inside another
# user-specified path
# ******************************************************************************


######################### USER-DEFINED VARIABLES ###############################

where="/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08/AH_data"

links_to=(
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0000/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0001/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0002/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0003/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0004/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0005/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0006/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0007/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0008/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0009/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0010/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0011/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0012/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0013/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0014/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0015/Miscellaneous"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08/output-0016/Miscellaneous"
)

links_names=(
    "output-0000"
    "output-0001"
    "output-0002"
    "output-0003"
    "output-0004"
    "output-0005"
    "output-0006"
    "output-0007"
    "output-0008"
    "output-0009"
    "output-0010"
    "output-0011"
    "output-0012"
    "output-0013"
    "output-0014"
    "output-0015"
    "output-0016"
)

################################################################################


# Sanity check
L=${#links_to[@]}
Ln=${#links_names[@]}
if [ $Ln != $L ]
then
    echo "ERROR: mismatching dimensions of input arrays ($L, $Ln)"
    exit 255
fi

cd $where

for ((l=0; l<$L; ++l))
do
    ln -s ${links_to[$l]} ${links_names[$l]}
done
