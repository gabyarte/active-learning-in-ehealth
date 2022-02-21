import spacy
from model.base import NER


class SpacyNER(NER):
    def __init__(self, tags):
        self.tags = tags

        self.nlp = spacy.blank('eHealth')

        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')

        for tag in tags:
            ner.add_label(tag)

    def fit(self, x_train, y_train, epochs=1, validation_data=None):
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            for epoch in range(epochs):
                losses = {}
                batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                print('Losses', losses) 