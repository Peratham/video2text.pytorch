#!/usr/bin/env bash

clear

case "$1" in
    0)
    echo "run vgg"
    CUDA_VISIBLE_DEVICES=0 python train.py --start_from='save/09220040_debug_msvd'
    ;;

    *)
    echo
    echo "No input"
    ;;
esac
