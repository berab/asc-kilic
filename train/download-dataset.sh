#!/bin/bash

for i in {1..21}
do
    wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.audio.$i.zip?download=1 -O audio_$i.zip
    
    unzip audio_$i.zip
    rm -f audio_$i.zip
done

for i in {1..13}
do
    wget https://zenodo.org/record/3685835/files/TAU-urban-acoustic-scenes-2020-3class-evaluation.audio.$i.zip?download=1 -O eval_$i.zip
    
    unzip eval_$i.zip
    rm -f eval_$i.zip
done

wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.doc.zip?download=1 -O doc.zip
wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.meta.zip?download=1 -O meta.zip

unzip meta.zip
unzip doc.zip
rm -f meta.zip doc.zip
