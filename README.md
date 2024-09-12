# Model Template - Deploy Gemma-2-9b-it using Inferless
Gemma-2-9b-it is an open-source AI language model developed by Google. It's designed to be efficient and lightweight, making it suitable for various natural language processing tasks on resource-constrained devices.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Add Your Hugging Face Access Token
Go into the `inferless.yaml` and replace `<YOUR_HUGGINGFACE_ACCESS_TOKEN>` with your hugging face access token. Make sure to check the repo is private to protect your hugging face token.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
      "inputs": [
        {
          "name": "prompt",
          "shape": [1],
          "data": ["What is deep learning?"],
          "datatype": "BYTES"
        },
        {
          "name": "temperature",
          "optional": true,
          "shape": [1],
          "data": [0.7],
          "datatype": "FP32"
        },
        {
          "name": "top_p",
          "optional": true,
          "shape": [1],
          "data": [0.1],
          "datatype": "FP32"
        },
        {
          "name": "top_k",
          "optional": true,
          "shape": [1],
          "data": [40],
          "datatype": "INT8"
        },
        {
          "name": "repetition_penalty",
          "optional": true,
          "shape": [1],
          "data": [1.18],
          "datatype": "FP32"
        },
        {
          "name": "max_new_tokens",
          "optional": true,
          "shape": [1],
          "data": [128],
          "datatype": "INT16"
        }
      ]
    }'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
def infer(self, inputs):
    prompt = inputs["prompt"]
    temperature = inputs.get("temperature", 0.7)
    top_p = inputs.get("top_p", 0.1)
    top_k = inputs.get("top_k", 40)
    repetition_penalty = inputs.get("repetition_penalty", 1.18)
    max_new_tokens = inputs.get("max_new_tokens", 256)
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self):
    self.model = None
```


For more information refer to the [Inferless docs](https://docs.inferless.com/).
