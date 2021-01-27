# Git

## Configure name, email
```
git config --global user.name "Raghav Mittal"
git config --global user.email raghav@some-domain.com
```

## Basic Stuff
Standard procedure:
```
git clone /path/to/repository
git add <filename>
git commit -m "Commit message"
git pull
git push
```

See what's up:
```
git status
git remote -v
git branch
git log
```


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


## Checking out a branch from a different remote
#### With One Remote
With Git versions ≥ 1.6.6, with only one remote, you can fetch and checkout. (Note: ```git checkout test``` will NOT work in modern git if you have multiple remotes
```
git fetch
git checkout <branch_name>
```

#### With >1 Remotes
Before you can start working locally on a remote branch, you need to fetch a branch. Fetch all of the remote branches, and then see the branches available for checkout with:
```
git fetch origin
git branch -v -a
```

With the remote branches in hand, you now need to check out the branch you are interested in, giving you a local working copy:
```
git checkout -b <branch_name> <remote_name>/<branch_name>
```
or the shorthand:
```
git checkout -t <remote_name>/<branch_name>
```


## Syncing Forks (Eg. origin/master with remote/master)
Use this [Fork-Branch-Git workflow](https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/)!
```
git remote add upstream https://github.com/udacity/deep-learning-v2-pytorch.git
git pull upstream master
git push origin master
```


