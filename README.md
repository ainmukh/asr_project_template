# ASR project: DeepSpeech

## May be not so obvious
```shell
git clone https://github.com/ainmukh/asr_project_template
cd asr_project_template
```

## Installation guide
```shell
pip install -q -r ./requirements.txt
```

## Get baseline and split
```shell
mkdir saved
wget https://www.dropbox.com/s/etbywu0xvvccwsl/indices.pth
wget https://www.dropbox.com/s/tvr7bq7kj7ct00k/model_base.pth
mv indices.pth model_base.pth
```
## Get all models
```shell
mkdir saved
wget https://www.dropbox.com/sh/dg7ofsb1xollq49/AACpI2Fk9pw9qtwmP5j8WxNUa
unzip AACpI2Fk9pw9qtwmP5j8WxNUa -d saved
```
