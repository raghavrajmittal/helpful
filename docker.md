# Docker


## Remove old containers and images (do this periodically)
To view all containers:
```
docker ps -a 
# which is equivalent to: docker container ls -a
```

To view all exited containers:
```
docker container ls -a -f status=exited
```

To remove a single container:
```
docker container rm <container_hash>
```

To remove all stopped containers:
```
docker container prune
```

To view all images (including intermediate ones that are cached):
```
docker image ls -a
```

To remove a single image:
```
docker image rm <image_hash>
```

To view all dangling images:
```
docker images -f dangling=true
```

To remove all dangling images:
```
docker image prune
```

To clean the whole system (stopped containers + unused networks + all dangling images + dangling build cache):
```
docker system prune
```
