class Evaluation:
    def __init__(self, true_positive,false_positive,true_negative,false_negative):
        self.true_positive =true_positive
        self.false_positive=false_positive
        self.true_negative=true_negative
        self.false_negative=false_negative

    def get_precision(self):
        return self.true_positive/(self.true_positive+self.false_positive)
    def get_recall(self):
        return self.true_positive/(self.true_positive+self.false_negative)
    def get_f_measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2*precision*recall / (precision+recall)


# How to Use
# import evaluation
# eval = evaluation.Evaluation(50,2,50,0)
# print(eval.get_recall())