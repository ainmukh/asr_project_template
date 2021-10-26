# ASR project: DeepSpeech

## Installation guide

```shell
pip install -r ./requirements.txt
pip install ctcdecoder==0.1.0
wget https://www.dropbox.com/sh/o67ylzg2pkdskx0/AAABb4RGYE1-5xlMarL4OLDta
unzip AAABb4RGYE1-5xlMarL4OLDta -d hw_asr/pretrained
mv hw_asr/pretrained/checkpoint.pth checkpoint.pth
```

## Usage
```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```
