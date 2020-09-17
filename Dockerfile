# set base image (host OS)
FROM python:3.5

# set the working directory in the container
WORKDIR /sorn

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
# COPY sorn/ .

# command to run on container start
CMD [ "python", "pip install sorn" ]