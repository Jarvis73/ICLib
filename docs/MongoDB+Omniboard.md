# Experiment Management

We utilize Sacred for managing experiments, and we use MongoDB and Omniboard for recording experiments. This doc is a simple tutorial to install MongoDB and Omniboard. 



## Install with Docker (Recommended)

-   Install Docker following the [instruction](https://docs.docker.com/engine/install/ubuntu/).
-   Install MongoDB

```
# Pull the image
docker pull mongo:4.4.4-bionic

# Create network (for Omniboard connection)
docker network create sacred

# Start a container.
docker run -p 7000:27017 -v /data/db:/data/db --name sacred_mongo --network sacred -d mongo:4.4.4-bionic
```

-   Install Omniboard
```
# Pull the image
docker pull vivekratnavel/omniboard

# Notice that 'ICLib' is the experiment name defined in main.py
docker run -p 17001:9000 --name omniboard --network sacred -d vivekratnavel/omniboard -m sacred_mongo:27017:ICLib
```

-   Open the address [http://localhost:17001](http://localhost:17001/)



## Install Manually

-   Download a proper version of [MongoDB](https://www.mongodb.com/download-center/community), and install with following commands.

```
curl -O https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-4.4.4.tgz
tar -zxvf mongodb-linux-x86_64-ubuntu1804-4.4.4.tgz

# Start MongoDB. Replace <path_to_mongodb> to the absolute path of mongodb
/<path_to_mongodb>/bin/mongod --dbpath /data/db --port 27107 --bind_ip_all
```

-   Install Nodejs

```
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
source　~/.bashrc
nvm install node
```

-   Install Omniboard

```
npm install -g omniboard

# Start Omniboard. Notice that 'ICLib' is the experiment name defined in main.py
PORT=17001 omniboard -m localhost:27107:ICLib
```

-   Open the address [http://localhost:17001](http://localhost:17001/)

