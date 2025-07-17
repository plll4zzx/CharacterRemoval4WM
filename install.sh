ENV_NAME="test_char"
PYTHON_VERSION="3.9"

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

conda init

. "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"
echo $CONDA_DEFAULT_ENV

sudo apt-get update
sudo apt-get install pkg-config
sudo apt-get install cmake
sudo apt install openjdk-17-jdk
sudo apt install tesseract-ocr

pip install Cmake
pip install torch==2.5.1
pip install transformers==4.49.0
pip install aiohttp==3.10.10
pip install flair==0.14.0
pip install textattack
pip install homoglyphs==2.0.4
pip install gensim==4.3.3
pip install rouge_score==0.1.2
pip install Levenshtein==0.26.1
pip install pytesseract==0.3.13
pip install textblob==0.19.0
pip install evaluate==0.4.3
pip install scipy==1.13.1
pip install scikit-learn==1.5.2
pip install seaborn==0.13.2
pip install pygad==3.3.1