Use this blueprint to deploy a QnA-generator API endpoint. To use this pretrained QnA-generator model, create a ready-to-use API-endpoint that is quickly integrated with your raw text data as input and is returned as pairs of questions and answers as output.

This inference blueprint’s model was trained using the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) [v1.0 dataset](https://huggingface.co/datasets/squad). To use custom QnA data according to your specific business, run this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/qna-training), which trains the model and establishes an endpoint based on the newly trained model.

Complete the following steps to deploy a QnA-generator API endpoint:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the dialog, select the relevant compute to deploy the API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   - Use the Try it Live section with any text passage to check the model.
   - Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

An API endpoint that returns the QnA pairs for input text has now been deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/qna-blueprint).
