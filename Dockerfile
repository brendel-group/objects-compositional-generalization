# Use NVIDIA's PyTorch image as the base
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Avoiding user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Add the deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Install a specific version of Python, e.g., Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Make Python 3.10 the default for python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set environment variables for Python to run more smoothly in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}:/home/jupyteruser/.local/lib/python3.10/site-packages"
    # for Jupyter to find the installed packages

# Update system packages and install essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    curl \
    bash-completion \
    build-essential \
    wget \
    git \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pipы
RUN pip install --upgrade pip

# Copy the requirements file from host to container
COPY requirements.txt ./

# Install dependencies from requirements.txt
RUN pip install jupyter && \
    pip install -r requirements.txt

# Set the working directory
WORKDIR /code

# Copy the rest of the project files (optional, based on original Dockerfile)
COPY . /code

# Create a non-root user and give permissions to the working directory
RUN useradd -m jupyteruser && \
    chown -R jupyteruser /code

# Switch to the non-root user
USER jupyteruser

# Expose the Jupyter port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]