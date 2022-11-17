# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import sys
import os
import fitz
import urllib.request
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
import nltk

try: #download nltk if not already installed
    nltk.download("punkt")
except:
    pass

scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(scripts_dir)

from pipelines import pipeline


tokenizer = ""
nlp = ""
folder = ""

#check if there exists a path for a new trained model, if so load the new trained model
if os.path.exists("/input/train/"):
    with open("/input/train/name.txt", "r") as f:
        folder = f.readline()

    tokenizer = AutoTokenizer.from_pretrained("/input/train/" + folder)
    nlp = pipeline("multitask-qa-qg", model="/input/train/" + folder)

#else load the default model from huggingface    
else:
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qa-qg-hl")
    nlp = pipeline("question-generation", model="valhalla/t5-base-qg-hl")


def predict(data):
    QnA = []
    #read the input text sent via API call
    input_text = data["context"]
    #check if the input text is a valid URL pointing a PDF or txt file, if so read the specific file otherwise process the raw string from API call
    try:
        input_text = extract_data(input_text, "tt")
    except:
        pass
    #add full stop at the end of the text if not already present to mark end
    if input_text[-1] != ".":
        input_text += "."
    encoded_input = tokenizer(input_text) #encode the entire text to get the total token size
    process = []
    to_loop = len(encoded_input["input_ids"]) // 512 + 1 #check the number of chunks we can make of 512 token size
    for i in range(to_loop):
        breakup = tokenizer.decode(encoded_input["input_ids"][:512]) #convert first 512 tokens to raw text.
        end_sentence = breakup.rfind(".") #find the last full stop in the text to find the end of the last complete sentence
        if end_sentence != -1:
            process.append(breakup[0 : end_sentence + 1]) #break the raw text at the last complete sentence and add it to the list
            input_text = input_text[end_sentence + 1 :] #take the remaining raw text
            encoded_input = tokenizer(input_text) #convert it into tokens again
        else:
            process.append(breakup) #if full stop not found add the entire text to the list
            input_text = input_text[len(breakup):] #take the remaining raw text
            encoded_input = tokenizer(input_text) #convert it into tokens again

    for textblock in process: #for each piece of raw text generating at max 512 token size generate QnA pairs.
        QnA.extend(nlp(textblock)) #add all QnA pairs to a list
    return {"prediction": QnA} #return the list


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# function to check if the link is valid url pointing to a pdf or txt
# it wil read pdf and txt files from a valid url and return a exception otherwise
def extract_data(download_url, filename):
    try:
        # if url is not from google drive and its pdf
        if download_url.find(".pdf") != -1:
            response = urllib.request.urlopen(download_url)
            file = open(filename + ".pdf", "wb")
            file.write(response.read())
            file.close()

            with fitz.open(filename + ".pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text

        # if url is not from google drive and its txt
        elif download_url.find(".txt") != -1:
            response = urllib.request.urlopen(download_url)
            file = open(filename + ".txt", "wb")
            file.write(response.read())
            file.close()
            with open(filename + ".txt", encoding="utf8") as f:
                text = f.read()
            return text

        else:
            # file is from google drive, now getting the extension of that file (.txt or .pdf)
            r = requests.get(download_url)
            soup = BeautifulSoup(r.content, features="lxml")
            for name in soup.findAll("title"):
                link = name.string
                name = link.replace(" - Google Drive", "")

            # if extenions is pdf
            if name.find(".pdf") != -1:
                download_url = download_url.split("/")[5]
                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()

                response = session.get(URL, params={"id": download_url}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {"id": id, "confirm": token}
                    response = session.get(URL, params=params, stream=True)
                filename = filename + ".pdf"
                save_response_content(response, filename)
                with fitz.open(filename) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                return text
            else:
                # if extentions is .txt
                download_url = download_url.split("/")[5]
                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()

                response = session.get(URL, params={"id": download_url}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {"id": id, "confirm": token}
                    response = session.get(URL, params=params, stream=True)
                filename = filename + ".txt"
                save_response_content(response, filename)
                with open(filename, encoding="utf8") as f:
                    text = f.read()
                return text
    except:
        raise Exception("URL not valid")
