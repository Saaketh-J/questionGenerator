from typing import Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import traceback
import pke
import string
from nltk.corpus import stopwords
import nltk
from flashtext import KeywordProcessor
import streamlit as st

# !pip install --quiet git+https://github.com/boudinfl/pke.git@dc4d5f21e0ffe64c4df93c46146d29d1c522476b
# !pip install --quiet flashtext==2.7

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
# !pip install --quiet transformers==4.5.0
# !pip install --quiet sentencepiece==0.1.95
# !pip install --quiet textwrap3==0.9.2
# !pip install --quiet nltk==3.2.5

st.title("Question Generation Bot")
st.subheader("Get practice questions for any piece of text that you're studying. Ex. A paragraph on Napoleon from your History textbook.")

st.markdown('#')


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " "+sent
    return final


def summarizer(text, model, tokenizer, device):
    text = text.strip().replace("\n", " ")
    text = "summarize: "+text
    max_len = 512
    encoding = tokenizer.encode_plus(
        text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             early_stopping=True,
                             num_beams=3,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             min_length=75,
                             max_length=300)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary


def get_nouns_multipartite(content):
    output = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)

        pos = {'PRRPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            output.append(val[0])
    except:
        output = []
        traceback.print_exc()

    return output


def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext)

    print("keywords unsummarized: ", keywords)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))
    print("keywords_found in summarized: ", keywords_found)

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:5]


def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


@st.cache(allow_output_mutation=True)
def load_model():
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_model = summary_model.to(device)

    question_model = T5ForConditionalGeneration.from_pretrained(
        'ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained(
        'ramsrigouthamg/t5_squad_v1')
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_model = question_model.to(device2)

    return summary_model, question_model, summary_tokenizer, question_tokenizer


with st.spinner('Loading Model'):
    summary_model, question_model, summary_tokenizer, question_tokenizer = load_model()


text = st.text_area("Enter text below.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if text:
    st.write('Questions:')

    with st.spinner('Generating....'):
        summarized_text = summarizer(
            text, summary_model, summary_tokenizer, device)
        imp_keywords = get_keywords(text, summarized_text)

        st.write(imp_keywords)

        questions = {}
        for answer in imp_keywords:
            ques = get_question(summarized_text, answer,
                                question_model, question_tokenizer)
            questions[ques] = answer

        st.write(questions)
