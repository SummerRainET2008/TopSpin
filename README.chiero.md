# Run in Chiero Cluster 

### Rules
To maintain a stable GPU cluster to guarantee the efficiency of most of the cluster users, approved by Doctor Chang, all users must obey these rules. Or, administrators, Tian Xia, have the right to reject you.
1. To become a Chiero cluster user, you must email the administrator to apply for an approval. Especially, you should clearly write 'I am willing to comply with the Chiero cluster rules'.
2. In principle, no specific cluster node would be assigned to a user constantly, except being approved for a special purpose with a definitive time frame.
3. Resouces are fair to all users. However, if your team contributes more machines to the cluster, you would have more privileges.
4. When GPUs are free, you can use as many as possible. However, if other users need to use it, you must obey rule 3 and the administrator's arrangement.
5. All users are subject to future updates of rules.

### Docker machines IPs 

Chiero cluster, so far, in colovore site, has FIVE 8-GPU servers, namely 5x8=40 cards. 

 1. **docker81**, ip=___10.10.10.81___, GPU 32G V100 x 8
 1. **docker88**, ip=___10.10.10.88___, GPU 48G rtx8000 x 8
 1. **docker89**, ip=___10.10.10.89___, GPU 48G rtx8000 x 8
 1. **docker90**, ip=___10.10.10.90___, GPU 48G a6000 x 8
 1. **docker91**, ip=___10.10.10.91___, GPU 48G a6000 x 8 
 1. **docker92**, ip=___10.10.10.92___, GPU 48G a6000 x 8

Python version **3.7** and **3.8** have passed test, yet **3.9** failed.

### Run in a docker machine
***Step 1***, login a jumping macine. 

Generate `id_rsa.pub`
```
# 1. >> ssh-keygen -t rsa
#   生成的过程中提示输入，直接回车，接受默认值就行了。
#   其中公共密钥保存在 ~/.ssh/id_rsa.pub， 私有密钥保存在 ~/.ssh/id_rsa
# 2. 然后改一下 .ssh 目录的权限，使用命令 "chmod 755 ~/.ssh"
# 3. 之后把这个密钥对中的公共密钥复制到你要访问的机器上去，并保存为~/.ssh/authorized_keys.
# 4. 设置权限: chmod 644 ~/.ssh/authorized_keys

```
Send to Tian Xia.
Then

```ssh chiero@172.20.2.185```
    
***Step 2***, login a docker machine.

```ssh root@10.10.10.{88, 89, 90, 91, 92}.```
    password: chiero123 

***Step 3***, set your data folder.

I recommend that you set your working directory in ```/NAS5/{your-group}/{your-pingan-account}```. The ```/NAS5``` has been mounted in every docker machine.
Then you can swith docker machine seamlessly to run on any empty node and available GPUs.

However, each docker also has a local storage in /data/{your-pingan-account}. You can set yourself. Note, do NOT use the local storage, unless you want to optimze the data loading speed to an extreme. 

***Step 4***, PAL_frame is always ready

The PAL_frame, maintained by Summer, is located in ```/NAS5/nlp/public/PAL_frame```. The path has been added into ```PYTHONPATH```, and you just import it in your code.

In PAL_frame/README.*, there are many documents from Tian Xia and Zijia Chen, introducing all functions, tricks of PAL_frame.

In PAL_frame/example, there are three examples from NLP, CV, ASR, showing a basic usage of PAL_frame.

I welcome you to contribute your fundermental algorithms and modules to PAL frame, and also welcome you to report bugs. 




