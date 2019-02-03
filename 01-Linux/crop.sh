#!/bin/bash
cd ~/Documents/Vision\ Artificial/mine/01-Linux/
rm -rf image_crop
mkdir -p image_crop/images/{train,val,test}
cd ~/Documents/Vision\ Artificial/mine/01-Linux/BSR/BSDS500/data/
ima=$(find images/ -name *jpg)
for im in ${ima[*]}
do
convert ./$im -gravity center -crop 250x250+0+0 ~/Documents/Vision\ Artificial/mine/01-Linux/image_crop/$im
done
