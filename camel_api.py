from fastapi import FastAPI
from pydantic import BaseModel
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tagger.default import DefaultTagger
from camel_tools.utils.dediac import dediac_ar
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# ASGI app variable
app = FastAPI()


@app.get("/")
def read_root():
    return {"camel_tools": "API for non-python applications"}


# data to be sent for analysis requests
class AnalysisRequest(BaseModel):
    text: str
    flag: str


# analysis endpoint
@app.post("/camel_tools")
def camel_analysis(req: AnalysisRequest):

    if req.flag == 'tokenizer':
        output = simple_word_tokenize(req.text)
    elif req.flag == 'tagger':
        mled = MLEDisambiguator.pretrained()
        tagger = DefaultTagger(mled, 'pos')
        output = tagger.tag(req.text.split())
    elif req.flag == 'disambig':
        mle = MLEDisambiguator.pretrained()
        disambig = mle.disambiguate(req.text.split())
        diacritized = [d.analyses[0].analysis['diac'] for d in disambig]
        output = ' '.join(diacritized)
    elif req.flag == 'dediac':
        output = dediac_ar(req.text)
    elif req.flag == 'root_stem':
        db = MorphologyDB.builtin_db()
        analyzer = Analyzer(db)
        analyses = analyzer.analyze(req.text)
        root = [r['root'] for r in analyses][0]
        stem = [s['stem'] for s in analyses][0]
        output = 'root: ' + root + ', ' + 'stem: ' + stem
    else:
        output = 'Please choose one: tokenizer, tagger, disambig, dediac, or root_stem'

    return {'output': output}
