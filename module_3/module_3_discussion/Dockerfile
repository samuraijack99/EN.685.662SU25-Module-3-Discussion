# Use a lightweight Ubuntu base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /workspace

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1 \
    && apt-get clean

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p /opt/anaconda && \
    rm anaconda.sh

ENV PATH="/opt/anaconda/bin:$PATH"

# Copy environment file and app files
COPY environment.yml .
COPY . .

# Create Conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Use conda in shell and run Streamlit
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["conda", "run", "-n", "myenv", "streamlit", "run", "test_app_student.py", "--server.port=8501", "--server.address=0.0.0.0"]
