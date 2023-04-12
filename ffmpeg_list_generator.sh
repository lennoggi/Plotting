# This script generates a list of files used by ffmpeg to generate a movie.
# Assuming the list of files is saved in a file called "ffmpeg_list.txt", the
# following command can be used to generate a movie from it:
#
#    ffmpeg -f concat -safe 0 -i ffmpeg_list.txt <outfilename>.mp4

# !/bin/bash


######################### USER-DEFINED VARIABLES ###############################

paths=(
    "/home/lorenzo/Downloads/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP_Frames"
    "/home/lorenzo/Downloads/CBD_handoff_IGM_McLachlan_Spinning_aligned08_Frames"
)

basenames=(
   "rho_xz_"
   "rho_b_xy_"
)


extensions=(
    ".png"
    ".png"
)

durations=(
    0.03
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
