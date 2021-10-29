# ASR project: DeepSpeech

## May be not so obvious
```shell
git clone https://github.com/ainmukh/asr_project_template
cd asr_project_template
```

## Installation guide

```shell
pip install -qqq -r ./requirements.txt
pip install ctcdecoder==0.1.0
wget https://www.dropbox.com/sh/o67ylzg2pkdskx0/AAABb4RGYE1-5xlMarL4OLDta
unzip AAABb4RGYE1-5xlMarL4OLDta -d hw_asr/pretrained
mv hw_asr/pretrained/checkpoint.pth checkpoint.pth
```

## Usage
```shell
python test.py \
   -c default_test_config.json \
   -r checkpoint.pth \
   -t test_data \
   -o test_result.json
```
