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
    "/home/lorenzo/Downloads/Comparisons/CBD_HydroDiskID_vs_handoff_Spinning_aligned08"
    ##"/home/lorenzo/Downloads/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/smallb2_xy_smallb2_xz_MediumSaturated"
    ##"/home/lorenzo/Downloads/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/rho_b_xy_rho_b_xz"
    ##"/home/lorenzo/Downloads/Comparisons/CBD_handoff_IGM_McLachlan_Spinning_aligned08_vs_PlusMinus08"
    ##
    ##"/home/lorenzo/Downloads/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP_Frames"
    ##"/home/lorenzo/Downloads/CBD_handoff_IGM_McLachlan_Spinning_aligned08_Frames"
)

basenames=(
    ##"smallb2_xy_smallb2_xz_"
    ##"rho_b_xy_rho_b_xz_"
    "rho_b_xy_rho_b_xy_"
    ##
    ##"rho_xz_"
    ##"rho_b_xy_"
)


extensions=(
    ".png"
    ##".png"
)

durations=(
    ##0.03
    0.04
)

list="/home/lorenzo/Downloads/ffmpeg_list.txt"

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
