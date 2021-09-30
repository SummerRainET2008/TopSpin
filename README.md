# Palframe
## install palframe
- Install from source code
 ```shell script
    git http://code.paic.com.cn/git/pal_frame.git
    cd pal_frame
    pip install -i http://mirrors.yun.paic.com.cn:4048/pypi/web/simple --trusted-host mirrors.yun.paic.com.cn -r requirements.txt
    python setup.py bdist_wheel 
    cd dist 
    pip install `xxx.whl`
```
- Install from github
```shell script
pip install -U git+http://code.paic.com.cn/git/pal_frame.git
```