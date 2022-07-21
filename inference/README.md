# Question Answer Generator
Question Answer generation is the task of generating pairs of questions and answers given some input text.
This library deploys a QnA maker capable of working with text from any domain and supports `English` language. 
The QnA maker is a two step process. It generates answers from a given input text based on the importance of words and placement of words in the sentence. It then takes the answers generated and input text and generates a question for each answer.

An example json response for some input text looks like below:
### Input Command

```
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"context": "Apart from counting words and characters, our online editor can help you to improve word choice and writing style, and, optionally, help you to detect grammar mistakes and plagiarism. To check word count, simply place your cursor into the text box above and start typing. You'll see the number of characters and words increase or decrease as you type, delete, and edit them."}'
```
### Response
```
{
    "prediction":
    {
        "prediction":
        [
            {
                "answer":"grammar mistakes and plagiarism",
                "question":"What can an online editor help you detect"
            },
            {   "answer":"increase or decrease",
                "question":"How do you see the number of characters and words as you type, delete, and edit them?"
            }
        ]
    }
}
```

### Reference
https://github.com/patil-suraj/question_generation 









