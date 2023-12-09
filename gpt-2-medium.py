import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# dir_name = 'gpt_raw'
# model_dir = "gpt2"

dir_name = 'result_0927/gpt2-medium'
model_dir = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, output_attentions=True)

def to_token(question, context):
    context = '<|im_start|>user ' + context 
    question = question + '<|im_end|> <|im_start|>assistant'
    q_token = tokenizer.tokenize(question)
    q_token = tokenizer.convert_tokens_to_ids(q_token)

    c_token = tokenizer.tokenize(context)
    c_token = tokenizer.convert_tokens_to_ids(c_token)

    s, e = len(q_token), min(len(q_token) + len(c_token), 2048)

    return (q_token + c_token)[:2048], s, e
    

if __name__ == '__main__':
    questions = open('qa.txt').readlines()
    questions = [x.strip() for x in questions]

    contexts = open('context.txt').readlines()
    contexts = [x.strip() for x in contexts]

    idx = 0
    for question in questions:
        for context in contexts:
            idx += 1
            print(idx, question)

            token, s, e = to_token(question, context)
            token = torch.tensor(token).unsqueeze(0)

            outputs = model(token)
            self_attentions = outputs['attentions']

            attentions = np.asarray([x.detach().numpy() for x in self_attentions])     # layer, batch_size, num_heads, sequence_length, sequence_length
            attentions = attentions[0][0].sum(0)
            attention_context = attentions[s:e, s:e]
            attention_cross = attentions[-s:, s:e]
            attention_question = attentions[-s:, -s:]
            attention_all = attentions

            from scipy.special import softmax
            attention_context, attention_cross = softmax(attention_context, 1), softmax(attention_cross, 1)
            attention_question, attention_all = softmax(attention_question, 1), softmax(attention_all, 1)


            with open('%s/self_%d.txt' % (dir_name, idx), 'w') as filein:
                print(' '.join(tokenizer.convert_ids_to_tokens(token[0][s:e])), file=filein)
                for line in attention_context:
                    print(' '.join([str(x) for x in line]), file=filein)
                print('', file=filein)
            
            with open('%s/cross_%d.txt' % (dir_name, idx), 'w') as filein:
                print(' '.join(tokenizer.convert_ids_to_tokens(token[0][0:s])), file=filein)
                print(' '.join(tokenizer.convert_ids_to_tokens(token[0][s:e])), file=filein)
                for line in attention_cross:
                    print(' '.join([str(x) for x in line]), file=filein)
                print('', file=filein)

            with open('%s/question_%d.txt' % (dir_name, idx), 'w') as filein:
                print(' '.join(tokenizer.convert_ids_to_tokens(token[0][0:s])), file=filein)
                for line in attention_question:
                    print(' '.join([str(x) for x in line]), file=filein)
                print('', file=filein)

            with open('%s/all_%d.txt' % (dir_name, idx), 'w') as filein:
                print(' '.join(tokenizer.convert_ids_to_tokens(token[0][:])), file=filein)
                for line in attention_all:
                    print(' '.join([str(x) for x in line]), file=filein)
                print('', file=filein)
                
            print(attention_context.shape, attention_cross.shape)

            context = '<|im_start|>user ' + context 
            question = question + '<|im_end|> <|im_start|>assistant'
            context = context + question

            input_ids = tokenizer.encode(context, return_tensors="pt")
            output = model.generate(input_ids, max_length=1000)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(output_text[output_text.find('Answer: '):])
