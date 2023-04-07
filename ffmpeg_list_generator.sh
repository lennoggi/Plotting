# This script generates a list of files used by ffmpeg to generate a movie

# !/bin/bash


######################### USER-DEFINED VARIABLES ###############################

paths=(
    "/home/lorenzo/Downloads/BBH_evolution_movie/CBD"
    "/home/lorenzo/Downloads/BBH_evolution_movie/CBD_End_ZoomingIn/Grid"
)

basenames=(
   "rho_xz_"
   "rho_"
)


extensions=(
    ".png"
    ".png"
)

durations=(
    0.02
    0.03
)

list="/home/lorenzo/Downloads/BBH_evolution_movie/ffmpeg_list.txt"

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
