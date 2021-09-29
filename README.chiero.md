# Run in Chiero Cluster 

### Basics

Chiero cluster, so far, in colovore site, has FIVE 8-GPU servers, namely 5x8=40 cards. 
Chiero comprises five dockers, and each one is located in one host machine. 

A docker IP is that its corresponding host IP minus 100. For example, a host machine is 10.10.10.190, then the docker IP is 10.10.10.90.

 1. **docker90**, ip=10.10.10.90, GPU 48G a6000 x 8
 1. **docker91**, ip=10.10.10.91, GPU 48G a6000 x 8 
 1. **docker92**, ip=10.10.10.92, GPU 48G a6000 x 8
 1. **docker89**, ip=10.10.10.89, GPU 48G rtx8000 x 8
 1. **docker88**, ip=10.10.10.88, GPU 48G rtx8000 x 8

Do **NOT** use ```192.168.1.*``` network.

Python version **3.7** and **3.8** have passed test, yet **3.9** failed.

### Run in a docker machine
***Step 1***, login a host machine,  

```ssh chiero@192.168.1.{188, 189, 190, 191, 192}.```
    password: chiero123
    
***Step 2***, login a docker machine, excep the one located in current host machine.

```ssh root@10.10.10.{88, 89, 90, 91, 92}.```
    password: chiero123 

For example, if your host machine is 190, then you can login 88, 89, 91, 92, except 90.

***Step 3***, set your data folder.

We make a folder `/home/chiero/chiero-data-link/` at 190 and use sshfs to mount this folder to each host. Please create a link of your data folder in this directory so you can access your data through acessing this directory.



