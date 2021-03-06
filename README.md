# gpt2-openvino (5x faster)
GPT2 using OpenVino speed up.

## Installation

If you have python-3.9 installed you will have to install python-3.7.9 for this project to work. Follow the instructions below when building for first time (verified build on MacOS):
```
brew install pyenv                             # for syncing multitple versions on the machine
pip3 install virtualenv                        # virtual-environment maker, can use any other package
pyenv install 3.7.9                            # install the specific version
pyenv local 3.7.9                              # set local (this folder) version to 3.7.9
export LOCAL_PY_VER_PATH=`pyenv which python3` # set path for convinience
echo $LOCAL_PY_VER_PATH                        # [opt.] to check the path
$LOCAL_PY_VER_PATH -m venv .                   # using the path above build a virtual environment in this folder
source bin/activate                            # activate the local env
pip3 install -r requirements.txt               # install run dependencies
```

When coming back to this project simply activate the virtualenv as and the rest will be ready for you:
```
source bin/activate
```

## ONNX Model

To get the model in the ONNX format first run the file `convert.py`, this should dump `gpt2.onnx` file.
```
python3 convert.py
```

## From ONNX to Openvino

For this you must first have openvino installed on your system. Download from [here](https://software.intel.com/en-us/openvino-toolkit). Now I have added most of the requirements in my `requirements.txt` file, however you should also install those for OpenVino. After that run the following commands to setup environment variables:
```
export OPENVINO_FOLDER="path/to/openvino_2021"
cd $OPENVINO_FOLDER/bin
source setupvars.sh
cd $OPENVINO_FOLDER/deployment_tools/model_optimizer
pip3 install install -r requirements.txt
pip3 install install -r requirements_onnx.txt
```

If everything works correctly you will see an output like this:
```
[setupvars.sh] OpenVINO environment initialized
```

Now come back to this repo, Openvino environment setup works correctly only if you are in the `openvino_2021/bin` folder. Now we run the script `mo_onnx.py`:
```
mo_onnx.py --help                              # to get meanings of arguments to be passed
mkdir full_precision half_precision            # full_precision is FP36 and other is FP16
mo_onnx.py --input_model gpt2.onnx \
--data_type=FP32/FP16 \
--output_dir=full_precision/half_precision
```

If everything works correctly you should see 3 files in `/fp32` folder:
```
gpt2.bin
gpt2.mapping
gpt2.xml
```

## Tests

#### Local Machine
To check if everything works fine run the script `run.py`. You should start seeing the outputs, the following is on the machine with following configuration:
```
MacBook Pro (13-inch, 2020, Four Thunderbolt 3 ports)
Processor: 2 GHz Quad-Core Intel Core i5
Memory:    16 GB 3733 MHz LPDDR4X
Graphics:  Intel Iris Plus Graphics 1536 MB
```
The performance results are as follows (`2x` boost):
```
----------------------------------------------------------------------
Loading Pytorch model
:: Pytorch inference in 0.59065s
----------------------------------------------------------------------
Creating Inference Engine...
Loading network
Loading IR to the plugin...
exec_net: <openvino.inference_engine.ie_api.ExecutableNetwork object at 0x12c531fb0>
:: OpenVino inference in 0.26206s
----------------------------------------------------------------------
```

In order to test generation capabilities you can pass `--g` flag and get the following results:
```
----------------------------------------------------------------------
Loading Pytorch model
Text shape: torch.Size([1, 127])
:: Pytorch inference in 0.46476s
----------------------------------------------------------------------
Testing generation
:: Pytorch generation took (40 steps): 17.663s
----------------------------------------------------------------------
Creating Inference Engine...
Loading network
Loading IR to the plugin...
exec_net: <openvino.inference_engine.ie_api.ExecutableNetwork object at 0x130aaffb0>
:: OpenVino inference in 0.23262s
----------------------------------------------------------------------
Testing generation
:: OpenVino generation took (40 steps): 6.220s
----------------------------------------------------------------------
```

#### Cloud Server

When running on AWS `c5.12xlarge` and batching the data to `128` samples in a batch we see larger performance increase.
```
----------------------------------------------------------------------
Loading Pytorch model
Pytorch inference in 3.55126s
----------------------------------------------------------------------
Creating Inference Engine...
Loading network
Loading IR to the plugin...
exec_net: <openvino.inference_engine.ie_api.ExecutableNetwork object at 0x12c531fb0>
----------------------------------------------------------------------
OpenVino inference in 0.78668s
----------------------------------------------------------------------
```
Which is a `5x` boost. Using OpenVino benchmarking tool we saw even more power throughput working at `134.29ms` of first inference and `17ms` as average processing time across `3522` runs. This is a massive **209x** speed improvement.

<img src="./image.png">

**This proves our hypothesis that larger CPU machines can take advantage of OpenVino's performance in a super-liear fashion.**
