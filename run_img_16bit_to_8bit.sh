for file in $(ls *.tif)
do
    echo "file is ${file}"
    python img_16bit_to_8bit.py --img_path=${file}
done
