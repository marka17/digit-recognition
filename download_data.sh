#!/usr/bin/env bash

python gfile.py -u "https://drive.google.com/file/d/1loObyNl2GiIsZr9Dp5SI1JeXnGUFrIkp" \
    -f "numbers.zip" \
    -d data/

unzip data/numbers.zip -d data/