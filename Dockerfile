FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /workspace/zilo-main

# System packages for OpenCV display, video IO, and basic utilities on Jetson.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Avoid conflicts between pip OpenCV wheels and Jetson's system OpenCV.
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true

# Install Python dependencies for edge / Jetson deployment.
COPY requirements-jetson.txt /tmp/requirements-jetson.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements-jetson.txt

# Copy project files after dependency installation to improve build caching.
COPY . .

ENV PYTHONPATH=/workspace/zilo-main
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Default to an interactive shell so users can export models or run webcam demos manually.
CMD ["bash"]
