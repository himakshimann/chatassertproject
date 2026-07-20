import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid deadlocks due to forks/parallelism
import json

import numpy as np
from numpy import dot
from numpy.linalg import norm

from transformers import AutoTokenizer, AutoModel
import torch

from utils.unixcoder import UniXcoder
from utils.file_util import read_file

from py4j.java_gateway import JavaGateway

# tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
# model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

def get_embeddings(code):
    global tokenizer, model

    # uniXcoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder("microsoft/unixcoder-base")
    tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, embeddings = model(source_ids)

    # # GraphCodeBERT
    # code_tokens = tokenizer.tokenize(code)
    # tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    # tokens_ids = [tokenizer.convert_tokens_to_ids(tokens)]

    # embeddings = model(torch.tensor(tokens_ids))[0][0][0].detach().numpy()

    return embeddings.detach()

def find_similar(target_class, target_name, target_code):
    print("sample_all.json not found. Skipping one-shot example retrieval.")
    return None

def test():
    file_path = "/Users/adminuser/Documents/Work/Experiment/ChatGPT/oragen-main/astvisitors/src/test/java/ncsusoftware/AnotherTest.java"
    test_code = '\n'.join(read_file(file_path, lo=53, hi=83))
    # test_code = "@Test\npublic void testSetType ( ) {\nString expected = \"STR\" ; \nactivityDefinition . setType ( expected ) ; \nString actual = activityDefinition . getType ( ) ; \nassertNotNull(actual);\n}"
    src_path = "/Users/adminuser/Documents/Work/Experiment/ChatGPT/oragen-main/astvisitors/src"

    print(find_similar('ActivityDefinitionTest', 'test', test_code))

