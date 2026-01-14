# python main.py \
# --cuda 0 \
# --model TI2V-5B \
# --size 480p \
# --arch DW \
# --precision baseline \
# --batch_size 1


CUDA="0,1,2,3,4,5,6,7"
SIZE="480p"
MODEL=TI2V-5B
BATCH_SIZE=8

ARCHS=(
  "NV"
  "DW"
)
PRECISIONS=(
  "nvfp4"
  "mxfp8"
  "baseline"
)

for ARCH in "${ARCHS[@]}"; do
    for PRECISION in "${PRECISIONS[@]}"; do
        echo "Running: model=${MODEL}, arch=${ARCH}, precision=${PRECISION}"
        python main.py \
            --cuda "${CUDA}" \
            --model "${MODEL}" \
            --size "${SIZE}" \
            --arch "${ARCH}" \
            --precision "${PRECISION}" \
            --batch_size ${BATCH_SIZE}
    done
done
