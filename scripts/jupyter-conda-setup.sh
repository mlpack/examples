#!/usr/bin/env bash

function cleanUp() {
    echo "**************"
    echo "Cleaning Up..."
    echo "**************"
    if [[ -f "./kernel.json" ]]; then
        rm kernel.json
    fi
    if [[ -f "./xeus-cling.hpp" ]]; then
        rm xeus-cling.hpp
    fi
    if [[ -f "./environment.yml" ]]; then
        rm environment.yml
    fi
    if [[ -d "${CONDA_PREFIX}/share/jupyter/lab" ]]; then
        jupyter lab clean 
    fi
}

function installDeps() {
    wget --version > /dev/null
    WGET_IS_AVIALABLE=$?
    if [[ $WGET_IS_AVAILABLE -ne 0 ]]; then
        conda install -c conda-forge wget
    fi
    git --version > /dev/null
    GIT_IS_AVAILABLE=$?
    if [[ $GIT_IS_AVAILABLE -ne 0 ]]; then
        conda install -c conda-forge git
    fi
    echo "*********************************"
    echo "Upgrading environment using mamba"
    echo "*********************************"
    if [[ -f "./environment.yml" ]]; then
        echo " "
    else
        wget -q https://raw.githubusercontent.com/mlpack/examples/master/binder/environment.yml
    fi
    mamba env update -f environment.yml
    npm install -g ijavascript 
    ijsinstall 
    echo "*********************************"
    echo "Installing SoS kernel"
    echo "*********************************"
    pip install sos-xeus-cling sos-javascript sos-python sos-r sos-julia 
    echo "*********************************"
    echo "Removing C++11 & C++17 Kernels"
    echo "*********************************"
    if [[ -d "${CONDA_PREFIX}/share/jupyter/kernels/xcpp11" ]]; then
        jupyter kernelspec remove -f xcpp11 
    fi
    if [[ -d "${CONDA_PREFIX}/share/jupyter/kernels/xcpp17" ]]; then
        jupyter kernelspec remove -f xcpp17 
    fi
    echo "*********************************"
    echo "Configuring lab environment"
    echo "*********************************"
    if [[ -f "./kernel.json" ]]; then
        echo " "
    else
        wget -q https://raw.githubusercontent.com/mlpack/examples/master/binder/kernel.json
    fi
    sed -i "4 s@/srv/conda/envs/notebook/bin/xcpp@$CONDA_PREFIX/bin/xcpp@g" kernel.json
    cp kernel.json $CONDA_PREFIX/share/jupyter/kernels/xcpp14/
    if [[ -f "./xeus-cling.hpp" ]]; then
        echo " "
    else
        wget -q https://raw.githubusercontent.com/mlpack/examples/master/binder/xeus-cling.hpp
    fi
    sed -i "s@/srv/conda/envs/notebook@$CONDA_PREFIX@g" xeus-cling.hpp
    cp xeus-cling.hpp $CONDA_PREFIX/include/mlpack/
    echo "*********************************"
    echo "Cloning mlpack/examples repo"
    echo "*********************************"
    if [[ -d "examples" ]]; then
        cd examples && git pull origin master && cd ..
    else
        git clone https://github.com/mlpack/examples.git 
    fi
    echo "*********************************"
    echo "Copying utilies to lab environment"
    echo "*********************************"
    cp -r ./examples/utils/* $CONDA_PREFIX/include/
    cleanUp
}

echo "***********************************"
echo "Installing mlpack local environment"
echo "***********************************"
if [[ $# -ne 1 ]]; then
    >&2 echo "ERROR: Invalid arguments, no environment name provided."
    >&2 echo "USAGE: install.sh <ENVIRONMENT_NAME>"
    exit 1
else
    envname=$1
    source $CONDA_PREFIX/bin/activate $envname

    if [[ $? -eq 0 ]]; then
        echo "***********************************"
        echo "Installing Dependencies in $envname"
        echo "***********************************"
        set -eE
        trap 'echo "ERROR setup unsuccessful" && cleanUp' ERR
        conda install -c conda-forge conda=4.11 python=3.9 mamba -y
        installDeps
    else
        echo "************************************"
        echo "Conda Environment: $envname is not found"
        echo "************************************"
        echo "Creating Conda Environment: $envname"
        echo "************************************"
        set -eE
        trap 'echo "ERROR setup unsuccessful, removing create conda environment" && conda remove -n ${envname} --all -y && cleanUp' ERR
        conda create -n $envname -c conda-forge conda=4.11 python=3.9 mamba -y
        if [[ $? -eq 0 ]]; then
            source $CONDA_PREFIX/bin/activate $envname
            installDeps
        fi
    fi
fi

echo "***************************"
echo "Environment Setup Completed"
echo "***************************"
echo "Usage: conda activate ${envname}"
echo "cd examples"
echo "jupyter lab"
