@echo on
(timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 5 100 > output\stn_run5.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 6 100 > output\cnn_run6.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 6 100 > output\stn_run6.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 6 100 > output\multistn_run6.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 7 100 > output\stn_run7.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 7 100 > output\multistn_run7.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 8 200 > output\cnn_run8.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 8 200 > output\stn_run8.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 9 200 > output\cnn_run9.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 9 200 > output\multistn_run9.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 10 200 > output\stn_run10.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 10 200 > output\multistn_run10.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 11 300 > output\cnn_run11.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 11 300 > output\stn_run11.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 11 300 > output\multistn_run11.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 12 300 > output\stn_run12.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 12 300 > output\multistn_run12.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 13 300 > output\stn_run13.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 14 400 > output\stn_run14.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 14 400 > output\multistn_run14.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 15 400 > output\cnn_run15.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 15 400 > output\stn_run15.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 15 400 > output\multistn_run15.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 16 400 > output\cnn_run16.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 16 400 > output\stn_run16.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 16 400 > output\multistn_run16.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 17 500 > output\cnn_run17.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 17 500 > output\stn_run17.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 17 500 > output\multistn_run17.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 18 500 > output\cnn_run18.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 18 500 > output\stn_run18.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 18 500 > output\multistn_run18.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_cnn_only.py 19 500 > output\cnn_run19.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_stn.py 19 500 > output\stn_run19.txt )^
& (timeout 5)^
& (python stn_code\cluttered_mnist_multistn.py 19 500 > output\multistn_run19.txt )

REM Template
REM & (timeout 5)^
REM & (python stn_code\cluttered_mnist_cnn_only.py 4 300 > output\cnn_run4.txt )^
REM & (timeout 5)^
REM & (python stn_code\cluttered_mnist_stn.py 4 300 > output\stn_run4.txt )^
REM & (timeout 5)^
REM & (python stn_code\cluttered_mnist_multistn.py 4 300 > output\multistn_run4.txt )^