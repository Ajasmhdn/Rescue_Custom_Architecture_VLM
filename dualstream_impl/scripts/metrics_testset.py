import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score

with open("outputs/test_predictions.json") as f:
    data = json.load(f)

refs = [d["ground_truth"] for d in data]
preds = [d["prediction"] for d in data]

bleu = sum(sentence_bleu([r.split()], p.split()) for r,p in zip(refs,preds))/len(refs)
print("BLEU:", bleu)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge = sum(scorer.score(r,p)['rougeL'].fmeasure for r,p in zip(refs,preds))/len(refs)
print("ROUGE-L:", rouge)

P,R,F1 = bert_score.score(preds, refs, lang="en")
print("BERTScore:", F1.mean().item())