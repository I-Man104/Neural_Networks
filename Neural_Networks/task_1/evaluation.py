class Evaluation:
    def __init__(self, actual,predicted):
        predicted = predicted.tolist()
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        for i, value in enumerate(actual):
            if actual[i] == predicted[i]:
                if actual[i] == 1:
                    self.true_pos += 1
                else:
                    self.true_neg += 1
            else:
                if actual[i] == 1:
                    self.false_neg += 1
                else:
                    self.false_pos += 1

        return self.true_pos, self.true_neg, self.false_pos, self.false_neg

    def get_precision(self):
        return self.true_positive/(self.true_positive+self.false_positive)
    def get_recall(self):
        return self.true_positive/(self.true_positive+self.false_negative)
    def get_f_measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2*precision*recall / (precision+recall)
