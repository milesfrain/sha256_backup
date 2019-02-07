Initial version cloned from:
https://github.com/moffa13/SHA256CUDA

Modified for efficiency, but can be cleaned-up further.

nvcc main.cu && ./a.out | tee -a leading.txt

tail -f -n +1 bits.txt | grep --line-buffered "aak" | tee leading.txt
