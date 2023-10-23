import subprocess
import nltk

repo_path = "https://github.com/bit-guber/py-readability-metrics/archive/refs/heads/master.zip"

def run():
    subprocess.run(["wget", repo_path]) 
    subprocess.run( [ "unzip", "master.zip" ] )

    subprocess.run(  [ "pip", "install", "-r", "requirements.txt" ]  )

    nltk.download(['stopwords', 'punkt',"wordnet","tagsets", "averaged_perceptron_tagger"])