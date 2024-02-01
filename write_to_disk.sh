#!/bin/bash

GB=$1
THREADS=$2
SIZE_IN_BYTES=$(($GB * 1024 * 1024 * 1024))
BYTES_PER_THREAD=$(($SIZE_IN_BYTES / $THREADS))
seq 0 $BYTES_PER_THREAD $(($SIZE_IN_BYTES - 1)) |
  parallel -k dd if=/dev/zero bs=64k iflag=fullblock,count_bytes oflag=direct count=$BYTES_PER_THREAD of=./write_{}

seq 0 $BYTES_PER_THREAD $(($SIZE_IN_BYTES - 1)) |
  parallel -k rm ./write_{}