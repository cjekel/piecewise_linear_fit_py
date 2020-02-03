#!/usr/bin/env bash
cp pwlf/pwlf.py pwlf/pwlfcp.py
sed -i 's/numpy/cupy/g' pwlf/pwlfcp.py
sed -i 's/np/cp/g' pwlf/pwlfcp.py
# sed -i 's/# import libraries/import numpy as np/g' pwlf/pwlfcp.py
sed -i 's/PiecewiseLinFit/PiecewiseLinFitCp/g' pwlf/pwlfcp.py 
sed -i 's/linalg.lstsq(A, self.y_data,/cp.linalg.lstsq(A, self.y_data)/g' pwlf/pwlfcp.py
sed -i '/                                       lapack_driver=self.lapack_driver)/d' pwlf/pwlfcp.py
sed -i 's/        # scale the sampling to my variable range/        mypop = cp.array(mypop)/g' pwlf/pwlfcp.py
# sed -i '/                else:\n                    ssr = ssr[0]\n/d' pwlf/pwlfcp.py
# perl -i -0777 -pe 's/                else:\n                    ssr = ssr[0]\n/himom\n\himom\n/igs' pwlf/pwlfcp.py
# sed -i 's/from scipy import linalg/from cupy import linalg/g' pwlf/pwlfcp.py

# sed -i 's/                else:\n /himom\n/g' pwlf/pwlfcp.py
# perl -i~ -0777 -pe 's/                else:\s*?ssr[0]/himom/s' pwlf/pwlfcp.py
# s/START.*?END/SINGLEWORD/s'
# sed -i '/                    ssr = ssr[0]  # easy sed flag/d' pwlf/pwlfcp.py
sed -i '/# easy sed flag/d' pwlf/pwlfcp.py
# sed -i 's/           x[i, :] = resx/           x[i, :] = cp.array(resx)/g' pwlf/pwlfcp.py
sed -i 's/        var = cp.sort(var)/        var = cp.sort(cp.array(var))/g' pwlf/pwlfcp.py
# sed -i '/# easy sed flag/d' pwlf/pwlfcp.py


sed -i 's/resf  # needs modification for cupy/resf[0]  # needs modification for cupy/g' pwlf/pwlfcp.py

# sed -i 's/A2inv = cp.abs(linalg.inv(cp.dot(A.T, A))).diagonal()/A2inv = cp.abs(cp.linalg.inv(cp.dot(A.T, A))).diagonal()/g' pwlf/pwlfcp.py

sed -i 's/linalg.inv/cp.linalg.inv/g' pwlf/pwlfcp.py
sed -i 's/linalg.solve/cp.linalg.solve/g' pwlf/pwlfcp.py

sed -i 's/stats.t.sf(cp.abs(t), df=n-k-1)/stats.t.sf(cp.asnumpy(cp.abs(t)), df=n-k-1)/g' pwlf/pwlfcp.py
sed -i 's/bounds=bounds/bounds=cp.asnumpy(bounds)/g' pwlf/pwlfcp.py
sed -i 's/var = cp.sort(res.x)/var = cp.sort(cp.array(res.x))/g'  pwlf/pwlfcp.py

cp tests/tests.py tests/testscp.py
sed -i 's/PiecewiseLinFit/PiecewiseLinFitCp/g' tests/testscp.py
sed -i 's/numpy/cupy/g' tests/testscp.py
sed -i 's/np/cp/g' tests/testscp.py
