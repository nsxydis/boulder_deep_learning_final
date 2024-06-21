'''
Purpose: Simple streamlit app to predict if a block of text
        was written by a person or AI.
'''

import polars as pl
import altair as alt
import streamlit as st
import string

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import functional as F
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pickle

# Set up on the gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    # Download the stopwords if you don't already have them...
    nltk.download('punkt', 'stopwords', quiet = True)

    # Get the text to analyze
    text = st.text_area("Text to analyze")
    
    # Wait for the user to provide text
    if text in [None, '']:
        return
    
    # Convert to a dataframe
    df = pl.DataFrame({'text' : [text]})

    # Remove stopwords
    df = df.with_columns(pl.col('text').map_elements(
        lambda x: removeCommon(x), 
        return_dtype = pl.Utf8,
        strategy = 'threading'
    ))

    # Plot the word distribution
    chart = plotWords(uniqueWords(df))
    chart.title = "Top word frequency of the sample text"
    st.altair_chart(chart)

    # Read in the vectorizer
    with open('model/tfidf_vectorizer_v1.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Transform the text
    dfEmbedding = tfidf.transform(df['text'])

    # Add the text tensor
    df = df.hstack(convertEmbeddingToTensor(dfEmbedding))

    # Dataset to test
    test = essayData(df) 

    # Model
    hidden_size = 128
    rnn = RNN(dfEmbedding.shape[1], hidden_size, output_size = 2)

    # Load the weights
    rnn.load_state_dict(torch.load('weights_v1.pt'))

    # Get the prediction
    results = inference(rnn, test, device)

    # Pull the probability
    prob = results['generated'][0]
    prediction = round(prob)

    # Display the results
    # prediction == 0: Not likely written by AI
    if prediction == 0:
        st.write('The model believes that this was not written by AI')

    # prediction == 1: Likely written by AI
    elif prediction == 1:
        st.write("The model believes this was written by AI")

    # Catch errors
    else:
        st.write('There was an error processing the data!')

    # Show the probability
    try:
        st.write(f'Predicted probability that this essay was written by AI: {round(prob * 100, 1)}%')
    except:
        pass

# Remove stopwords from the data and check again
def removeCommon(text):
    '''Removes common words & punctuation from text and returns the data'''
    n = 0
    while True:
        n += 1
        if n % 10 == 0: print(f"Attempt {n}")

        try:
            # NOTE: Hyphens are allowed for hyphenated words
            common = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            filtered = ""
            for word in tokens:
                if word not in common and (word.replace('-', "").isalnum()):
                    filtered += f" {word}"
            return filtered
        except:
            pass

def uniqueWords(df):
    '''Get all the unique words and counts from the given dataframe'''
    words = {}
    count = 0   # We'll keep count of the number of essays to normalize our results

    for n in range(len(df)):
        count += 1

        # Get the essay text
        essay = df['text'][n]

        # Remove punctuation
        essay = essay.translate(str.maketrans('', '', string.punctuation))

        # Remove capitalization
        essay = essay.lower()

        # Remove newlines with spaces
        essay = essay.replace('\n', ' ')

        # Split by word
        for word in essay.split(' '):
            if word in ['', ' ']:
                pass
            elif word in words:
                words[word] += 1
            else:
                words[word] = 1

    words = pl.DataFrame(words).transpose(include_header = True)
    words = words.rename({
        'column'    : 'word',
        'column_0'  : 'frequency'
    })

    # Normalize based on the number of essays
    words = words.with_columns(
        pl.col('frequency') / count
    )
    return words

def plotWords(df, freq = 10):
    '''Plot the distribution of the 10 most frequent words'''
    # Get the top 10 words
    df = df.sort(by = 'frequency', descending = True)[:freq]

    # Since we want to maintain the sort order
    order = df['word'].unique(maintain_order = True).to_list()

    # Make a chart
    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        alt.X('word', sort = order),
        alt.Y('frequency'),
        alt.Color('frequency', scale = alt.Scale(scheme = 'turbo')),
        tooltip = [
            'word',
            alt.Tooltip('frequency', title = 'Average frequency in an essay', format = '.1f')
        ]
    )
    return chart

def convertEmbeddingToTensor(embedding):
    data = {'tensor' : []}
    for row in embedding:
        t = torch.tensor(row.toarray(), dtype = torch.float32, device = device)
        data['tensor'].append(t)

    return pl.DataFrame(data)

# Dataset Class
class essayData(Dataset):
    def __init__(self, df):
        '''Process data from the given dataframe'''
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, n):
        '''Open the nth tensor'''
        # try:
        #     label = self.df['generated'][n]
        # except:
        #     label = 0
        label = None
        line = self.df['tensor'][n]
        line.to(device)

        return line, label

# Model
torch.manual_seed(42)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
 
    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device = device)

def inference(model, dataset, device):
    '''Process the data and predict, returns a polars dataframe of the results'''
    model = model.to(device)
    pred = {'generated' : []}
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i][0]
            # Init
            hidden = model.initHidden()

            # Get and store the predictions
            for i in range(x.size()[0]):
                output, hidden = model(x, hidden)

            # Get the probability and label
            prob, label = torch.topk(output, 2)
            
            # Get the index for the generated label
            index = 0 if label[0][0].item() == 1 else 1

            prob = nn.functional.softmax(prob, dim = 1)

            genProb = round(prob[0][index].item(), 2)
            pred['generated'].append(genProb)

    # Join the predictions to the original data
    df = dataset.df.hstack(pl.DataFrame(pred))

    return df

if __name__ == '__main__':
    main()