nvcc main.cu && ./a.out | tee -a leading.txt

tail -f -n +1 bits.txt | grep --line-buffered "aak" | tee leading.txt
