#! /bin/bash

if [ $# -ne 1 ];then
    echo "wrong"
    exit -1
fi

file=$1

file1=val.txt
file2=test.txt
file3=train.txt

n=150

./sampling.py $file $n > $file1
awk -F'\t' 'NR==FNR{a[$7]=1;next}{if(!($7 in a)) print}' $file1 $file > t1
./sampling.py t1 $n > $file2
awk -F'\t' 'NR==FNR{a[$7]=1;next}{if(!($7 in a)) print}' $file2 t1 > $file3

rm t1
