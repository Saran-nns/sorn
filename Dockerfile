# set base image
FROM python:3.8

# set the working directory in the scontainer
WORKDIR /sorn

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY sorn/ .

# command to run on container start
CMD [ "python", "pip install sorn" ]