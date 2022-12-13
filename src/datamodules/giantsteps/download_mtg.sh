#!/bin/bash -

# based on the script and annotations at https://github.com/GiantSteps/giantsteps-mtg-key-dataset

LABEL_FILES=./labels/*
BASEURL=http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/
AUDIOPATH=./audio/

handle_label() {
  label_file=${1}
  label=$(cat "$label_file")
  filename=$(basename "$label_file")
  mp3filename="${filename%.*}".mp3
  mp3url=${BASEURL}${mp3filename}
  audio_out_path=${AUDIOPATH}$label
  audio_out_path=$(echo $audio_out_path | sed 's/[[:space:]]/-/g')
  audio_out_path=$(echo $audio_out_path | sed 's/^([^-]*-[^-]*)-.*/\$1/')
  mkdir -p ${audio_out_path}

  audio_out_path=$audio_out_path/$mp3filename
  download_file "$audio_out_path" "$mp3url"
}

download_file() {
  curl -o "$1" "$2"
}

count=0

printf "\n"

# download labels
wget https://github.com/GiantSteps/giantsteps-mtg-key-dataset/archive/refs/heads/master.zip
unzip -j master.zip 'giantsteps-mtg-key-dataset-master/annotations/key/*' -d labels

for label_file in $LABEL_FILES; do
  handle_label ${label_file}
  ((count++))
done

rm ./master.zip
rm ./labels

printf "Downloaded: ${count} files\n"
