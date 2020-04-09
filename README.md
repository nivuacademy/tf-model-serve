```console

tensorflow_model_server --model_base_path=/mnt/e/git/tf-model-serve/mobile_v2 --rest_api_port=9000 --model_name=ImageClassifier

cd flask_Server
flask run --host=0.0.0.0
```

Test URL, Hello
http://localhost:5000/hello

Predict URL, Image
http://localhost:5000/imageclassifier/predict/
