## Install TensorFlow Server

```console

echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

sudo apt update

sudo apt-get install tensorflow-model-server

```

## Serve the Model

```console
tensorflow_model_server --model_base_path=/mnt/e/git/tf-model-serve/mobile_v2 --rest_api_port=9000 --model_name=ImageClassifier

cd flask_Server
flask run --host=0.0.0.0
```

Test URL, Hello
http://localhost:5000/hello

Predict URL, Image
http://localhost:5000/imageclassifier/predict/

tensorflow_model_server --model_base_path=/mnt/e/git/tf-model-serve/flower --rest_api_port=9000 --model_name=ImageClassifier
