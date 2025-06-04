#!/bin/bash

set -e  # Exit on any error

# 0. Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential python3-dev libopenblas-dev

# 1. Create Python virtual environment
echo "Creating Python virtual environment..."
sudo apt install python3.10-venv
python3.10 -m venv anygrasp_env
source anygrasp_env/bin/activate

# 2. Install torch, ninja, numpy
echo "Installing PyTorch, ninja, and numpy..."
pip install torch==2.0.1 ninja numpy

# 3. Install MinkowskiEngine (NVIDIA version)
iecho "Installing MinkowskiEngine from NVIDIA..."
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

# 4. Clone anygrasp_sdk repository
echo "Cloning anygrasp_sdk..."
git clone https://github.com/dimensionalOS/anygrasp_sdk
cd anygrasp_sdk

# 5. Install Python requirements
echo "Installing requirements..."
pip install -r requirements.txt

# 6. Install pointnet
echo "Installing pointnet..."
cd pointnet
python setup.py install
cd ..

# 7. Copy Python 3.10 specific shared object files
echo "Copying shared object files for Python 3.10..."
cp gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so gsnet.so
cp license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so lib_cxx.so

# 9. Fix OpenSSL / libcrypto issues
#echo "Fixing OpenSSL libcrypto errors..."
#mkdir -p $HOME/opt && cd $HOME/opt
#wget https://www.openssl.org/source/openssl-1.1.1o.tar.gz
#tar -zxvf openssl-1.1.1o.tar.gz
#cd openssl-1.1.1o
#./config && make && make test

#mkdir -p $HOME/opt/lib
#cp libcrypto.so.1.1 libssl.so.1.1 $HOME/opt/lib/

export LD_LIBRARY_PATH=$HOME/opt/lib:$LD_LIBRARY_PATH
cd $OLDPWD  # Go back to working directory

# Final instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "👉 Next steps:"
echo "  - Unzip your license files into anygrasp_sdk/grasp_detection/"
echo "  - Put .rar weight files into anygrasp_sdk/log/"
echo ""
echo "💡 To run the demo:"
echo "  source anygrasp_env/bin/activate"
echo "  cd anygrasp_sdk"
echo "  python demo.py --checkpoint_path log/checkpoint_detection.tar --top_down_grasp --debug"

