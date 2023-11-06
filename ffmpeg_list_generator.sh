#!/bin/bash

set -e

# ******************************************************************************
# This script generates a list of files used by ffmpeg to generate a movie.
# Assuming the list of files is saved in a file called "ffmpeg_list.txt", the
# following command can be used to generate a movie from it:
#
#    ffmpeg -f concat -safe 0 -i ffmpeg_list.txt -c:v libx264 -crf 17 -preset veryslow -tune film <outfilename>.mp4
#
# See the H.264 encoding tutorial at
# https://trac.ffmpeg.org/wiki/Encode/H.264
# ******************************************************************************



######################### USER-DEFINED VARIABLES ###############################

paths=(
    "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08/rho_b_xy_rho_b_xz_minidisk_disruption"
)

basenames=(
    "rho_b_xy_rho_b_xz_"
)


extensions=(
    ".png"
)

durations=(
    ##0.03
    0.1
)

list="/home1/07825/lennoggi/ffmpeg_list.txt"

################################################################################


# Sanity check
L=${#paths[@]}
Lb=${#basenames[@]}
Le=${#extensions[@]}
Ld=${#durations[@]}

if [ $Lb != $L ] || [ $Le != $L ] || [ $Ld != $L ]
then
    echo "ERROR: mismatching dimensions of input arrays"
    exit 255
fi


for ((l=0; l<$L; ++l))
do
    path=${paths[$l]}
    path+="/"
    path+=${basenames[$l]}
    path+="*"

    for file in $path
    do
        echo "file '$file'" >> $list
        echo "duration ${durations[$l]}" >> $list
    done
done
