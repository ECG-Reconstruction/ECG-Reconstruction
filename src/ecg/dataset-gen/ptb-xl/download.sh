#!/bin/bash

if [[ "$#" -eq 0 ]]; then
    # By default, save files to a `downloads` directory next to this script.
    save_dir="$(cd -P -- "$(dirname -- "$0")" && pwd -P)/downloads"
elif [[ "$#" -eq 1 ]]; then
    # Use a custom save directory if provided.
    save_dir="$(realpath -- "$1")"
else
    echo "Too many command-line arguments: $#"
    exit 1
fi

echo "Files will be saved to '$save_dir'"
# Create the save directory if it does not exist.
mkdir -p -- "$save_dir"

# Download the ZIP file and unzip it.
filename="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
wget -P "$save_dir" -- "https://physionet.org/static/published-projects/ptb-xl/$filename.zip"

# Move the unzipped files to the parent directory.
unzip -d "$save_dir" -- "$save_dir/$filename"
mv $save_dir/$filename/* $save_dir
rmdir $save_dir/$filename
