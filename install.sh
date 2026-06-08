
conda create -n mib python==3.10
pip install .
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128/
pip install openpyxl
pip install transformer-lens