Using an extensive dataset, this blueprint retrains a QnA model to enhance its performance. This blueprint also establishes an endpoint that returns QnA pairs for input text based on the newly trained model.

The Prepare library task prepares the input data for training by converting it into PyTorch datasets and splitting it into train and development data. The input data format for the Training task is SQuAD v1.0 in the form of a single JSON file. This blueprint trains the QnA model and deploys a fine-tuned model that can be used for inference using API calls.

Complete the following steps to train and deploy a QnA-generator model and an API endpoint:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **Prepare** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `data_source ` − Value: provide the path to the data folder including the S3 prefix
     - `/input/s3_connector/qna_data/trainfiltered.json` − ensure the path adheres to this format
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `--num_train_epochs` − Value: set the number of times the model trains on the dataset
     - Key: `--output_dir` − Value: provide the output folder to store the final trained model
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, producing a trained QnA-generator model and deploying it as a new API endpoint.
5. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   - Use the Try it Live section with any text passage to check the model.
   - Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

A QnA model and an API endpoint, which returns the QnA pairs for input text, have now been retrained and deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/qna-blueprint).
