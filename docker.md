# Docker


## Using Plain Docker

#### Components
- _Dockerfile_: Define steps to build an application. Each step `RUN`, `COPY` etc. creates a new layer

#### Commands
```
docker build .
```
Build the image. The `.` at the end provides the context

```
docker build -f Dockerfile-example -t python-flask .
```
The `-f` option is used to specify which dockerfile it looks at. By default (if not specified), it searches for the dockerfile named `Dockerfile` (capital D). The `-t` option is used to rename the created image (by default, it has a really long hashed name)

```
docker run python-flask
```
```
docker run -v $(pwd -P):./workarea python-flask
```
The `-v` option can be used to mount a volume in the container.




## Using the docker-compose toolset

#### Components
- _docker-compose.yaml_: File defining the docker image, services, volumes, network configs, runtime settings etc

#### Commands
```
docker-compose build
```
Build the image from the context given in the `docker-compose.yaml` file. Usually call it from the directory that contains both the `Dockerfile` as well as `docker-compose.yaml` files.

```
docker-compose up
```
Creates a running service container, mounths the volumes, creates a private network (and manages the DNS on that private network).

```
docker-compose down
```
Shuts down all services.

```
docker-compose exec <svc_name> bash
```
Connects to the service _svc-name_ that was started using `docker-compose up` and opens a bash shell.

```
docker-compose ps
```
List the running containers.

```
docker-compose run --rm <svc_name>
```
Run only a new container. The `-rm` option closes the container after execution.





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
