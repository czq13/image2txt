# coding=utf-8
from evalcap.bleu.bleu import Bleu
from evalcap.meteor.meteor import Meteor
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider

if __name__ == '__main__':

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    sentence1 = "大街的马路上有一个路标指向牌"
    sentence2 = "道路上有很多绿色的标志牌"

    gts={1:[sentence1]}
    res={1:[sentence2]}
    for scorer, method in scorers:
        print 'computing %s score...' % (scorer.method())
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print "%s: %0.3f" % (m, sc)
        else:
            print "%s: %0.3f" % (method, score)