
# Load package on memory 
import os, os, sys, warnings, json

if not os.path.isdir( "py-readability-metrics-master" ): # modified package ( removed each text require 100 words )
    import setup  # this requires download script
    setup.run()

sys.path.append( "py-readability-metrics-master" ) # added on directory
import pandas as pd, nltk, numpy as np
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize  
from collections import Counter
from readability import Readability
from pandarallel import pandarallel
from xgboost.sklearn import XGBRegressor



# Initial package setup 
warnings.filterwarnings("ignore")
pandarallel.initialize(nb_workers = os.cpu_count(),progress_bar=False)
lem = nltk.WordNetLemmatizer()
spellcheck = SpellChecker()
STOP_WORDS = set(stopwords.words('english'))

config = None
with open( "config.json", "r" ) as f:
    config=json.load(f )


# define functions for preprocess
def get_readability_features( text, names ):
    text = Readability(text)
    stats = text.statistics() 
    stats = [stats[ "avg_syllables_per_word" ], stats[ "avg_words_per_sentence" ],stats[ "num_polysyllabic_words" ]]
    for m in names[3:]:
        stats.append( getattr(text, m)().score )
    return stats

def get_data( file_name ):
    data = pd.read_csv( f"summaries_{file_name}.csv" )
    prompt = pd.read_csv( f"prompts_{file_name}.csv" )

    def preprocess_lem(word):
        for pos_type in [ "a", "n", "s", "r", "v" ]: 
            word = lem.lemmatize(word, pos = pos_type)
        return word
    
    rf_names = ["avg_syll", "avg_words", "n_polysyll", "ari", "coleman_liau", "flesch", "smog", "spache"]
    data[ ["text_"+ x for x in rf_names ] ] = data.parallel_apply( lambda x : get_readability_features( x["text"],rf_names ), axis = 1, result_type="expand" )
    
    rf_names = ["avg_syll", "avg_words", "n_polysyll", "coleman_liau", "smog"]
    prompt[ ["prompt_text_"+ x for x in rf_names ] ] = prompt.parallel_apply( lambda x : get_readability_features( x["prompt_text"],rf_names ), axis = 1, result_type="expand" )

    # preprocess
    data['text']  = data['text'].str.replace( r"\r|\n|\t"," ", regex = True ).str.lower()
    prompt['prompt_text'] = prompt['prompt_text'].str.replace( r"\r|\t|\n", " ", regex = True ).str.lower()
    prompt['prompt_question']  = prompt['prompt_question'].str.replace( r"\r|\n|\t"," ", regex = True ).str.lower()
    prompt['prompt_title'] = prompt['prompt_title'].str.replace( r"\r|\t|\n", " ", regex = True ).str.lower()

    for x in prompt.prompt_text.values:
        spellcheck.word_frequency.load_words( [ preprocess_lem(_) for _ in word_tokenize(x) if _.isalpha() ] )
    for x in prompt.prompt_question.values:
        spellcheck.word_frequency.load_words( [ preprocess_lem(_) for _ in word_tokenize(x) if _.isalpha() ] )

    def keyword_extract( df, col ):
        df[f"{col}_keywords"] = df[col].parallel_apply( lambda x : " ".join( [ preprocess_lem(_) for _ in word_tokenize(x) if _.isalpha() and _ not in STOP_WORDS ] )  )
    keyword_extract(data, "text")
    keyword_extract(prompt, "prompt_question")
    keyword_extract(prompt, "prompt_text")
    keyword_extract(prompt, "prompt_title")

    rf_names = ["avg_syll", "avg_words", "n_polysyll", "ari", "linsear_write"]
    data[ ["text_keywords_"+ x for x in rf_names ] ] = data.parallel_apply( lambda x : get_readability_features( x["text_keywords"],rf_names ), axis = 1, result_type="expand" )
    data.rename(columns = { "text_keywords_linsear_write":"text_keywords_linear_write" },inplace=True)
        # text_length 
    data["word_count"] = data['text'].parallel_apply(lambda x :len(x.split()))

    # keywords 
    data[ "keyword_count" ] = data['text_keywords'].parallel_apply( lambda x : len(x.split() ) )

    # misspelled_word_count 
    data['misspelled_word_count'] = data['text_keywords'].parallel_apply( lambda x : len(spellcheck.unknown( x.split() )) )
    # misspelled_word_ration
    data['misspelled_word_ratio'] = data['misspelled_word_count'] / data["keyword_count"]

    data = pd.merge(data, prompt, on='prompt_id')
    rf_names =["avg_syll", "avg_words", "n_polysyll", "coleman_liau", "smog" ]
    data[ "diff_pure_smog" ] = data['text_smog'].sub( data['prompt_text_smog'], fill_value = 0 ).abs()
    data[ "diff_pure_coleman_liau" ] = data['text_coleman_liau'].sub( data['prompt_text_coleman_liau'], fill_value = 0 ).abs()
    for x in rf_names:
        del data[ "prompt_text_"+x ]
        
    data['title_present'] = data.parallel_apply( lambda x : x["prompt_title"] in x['text'], axis= 1 ).astype( np.int8)

    data['stopword_count']= data["word_count"] - data['keyword_count']

    # overlapping_word_count
    def word_overlap_ratio(row, prompt, summary,   stop_words=STOP_WORDS):
        prompt_words = Counter(row[prompt].split())
        summary_words = Counter(row[summary].split())
        intersection = set(prompt_words.keys()).intersection(summary_words.keys())
        # initial mean
        total = []
        for x in intersection:
            total.append((summary_words[x]- prompt_words[x]) )

        total = np.array(total)
        return len(intersection), total.mean() if total.sum() > 0 else 0

    data[['overlapping_word_count', "overlapping_word_mean"]] = data.parallel_apply( lambda x :  word_overlap_ratio(x, "prompt_text_keywords", "text_keywords") ,axis = 1, result_type = "expand")

    #overlapping_ratio
    data['overlapping_word_ratio'] = data['overlapping_word_count'] / data['keyword_count']
    data['overlapping_word_mean_ratio'] = data['overlapping_word_mean'] / data['keyword_count']


    def ngram_co_occurrence(row,prompt,summary, n):
        # Tokenize the original text and summary into words
        original_tokens = row[prompt].split()
        summary_tokens = row[summary].split()

        # Generate n-grams for the original text and summary
        original_ngrams = set(ngrams(original_tokens, n))
        summary_ngrams = set(ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    n = 3
    for i in range( 2, n+1 ):
        data[f'{i}_gram_overlap_count'] = data.parallel_apply(ngram_co_occurrence, args=("prompt_text_keywords","text_keywords",i,), axis=1 )

        data[f"{i}_gram_overlap_ratio"] = data[f'{i}_gram_overlap_count'] / (data['keyword_count']/i)

    def detach_overlapping_words( row, prompt, summary ):
        summary_set = set(row[summary].split())
        prompt_set = set(row[prompt].split())
        intersection = summary_set.intersection(prompt_set)
        return " ".join( [ x for x in row[prompt].split() if x not in intersection ] )," ".join( [ x for x in row[summary].split() if x not in intersection ] )

    data[ [ "withdraw_prompt_text_keywords", "withdraw_text_keywords" ]  ] = data.parallel_apply( lambda x : detach_overlapping_words(x, "prompt_text_keywords", "text_keywords"), axis =1 , result_type = "expand" )

    data[ "withdraw_prompt_text_keywords" ] = data['withdraw_prompt_text_keywords'].parallel_apply(lambda x: " ".join( set(x.split()) ))
    data[ "withdraw_text_keywords" ] = data['withdraw_text_keywords'].parallel_apply(lambda x: " ".join( set(x.split()) ))
    data['withdraw_word_count'] = data['withdraw_text_keywords'].parallel_apply( lambda x: len(x.split()) )

    data['withdraw_prompt_text_keywords_ratio'] = data['withdraw_word_count'] / data['withdraw_prompt_text_keywords'].apply(lambda x: len(x.split()))
    data['experiment'] = data[ "withdraw_word_count" ] - data['misspelled_word_count']

    return data.replace( [-np.inf, np.nan, np.inf], 0 )