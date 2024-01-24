from simplenlg.lexicon import Lexicon
from simplenlg.framework import NLGFactory
from simplenlg.realiser import Realiser
from simplenlg.features import Feature, Tense

lexicon = Lexicon.getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)
p = nlgFactory.createClause()
p.setSubject("Developer")
p.setVerb("assign")
objectNP = nlgFactory.createNounPhrase("the", "bug")
p.setObject(objectNP)
p.setFeature(Feature.MODAL, "must")

p.setFeature(Feature.TENSE, Tense.PRESENT)

print(realiser.realiseSentence(p))
