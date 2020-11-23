# Git


## Creating a new local branch and pushing to remote

First, create a new local branch and check it out
```
git checkout -b <branch-name>
```



The remote branch is automatically created when you push it to the remote server. In most cases, the remote-name is 'origin'
```
git push <remote-name> <branch-name>
```



The formal format is listed below. But when you omit one, it assumes both branch names are the same. Having said this, as a word of caution, do not make the critical mistake of specifying only :remote-branch-name (with the colon), or the remote branch will be deleted!
```
git push <remote-name> <local-branch-name>:<remote-branch-name>
```  


The --set-upstream option sets up an upstream branch, so that a subsequent git pull will know what to do:
```
git push --set-upstream <remote-name> <local-branch-name>
```  

