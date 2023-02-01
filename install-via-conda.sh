if ! [ -x "$(command -v conda)" ]; then
    echo 'Error: conda is not installed.' >&2
    read -r -p "Proceed to install conda before installing the minimal-ml-template prerequesities? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p $HOME/miniconda
            source $HOME/miniconda/etc/profile.d/conda.sh
            conda init bash
            conda init fish
            source ~/.bashrc
            ;;
        *)
            echo "Will not install conda. Exiting installation of prerequesities. Please install conda manually before retrying."
            ;;
    esac
    
fi


conda create -n minimal-ml-template python=3.10 -y
conda activate minimal-ml-template 
conda install -c conda-forge mamba -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
mamba install -c conda-forge google-cloud-sdk -y
mamba install -c anaconda-platform kubectl -y
mamba install -c conda-forge timm accelerate datasets transformers -y
mamba install -c conda-forge orjson -y

echo y | pip install git+https://github.com/BayesWatch/bwatchcompute@main
echo y | pip install hydra_zen wandb

git config --global credential.helper store

echo y | pip install -e .