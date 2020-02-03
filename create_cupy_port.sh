#!/usr/bin/env bash
cp pwlf/pwlf.py pwlf/pwlfcp.py
sed -i 's/numpy/cupy/g' pwlf/pwlfcp.py
sed -i 's/np/cp/g' pwlf/pwlfcp.py