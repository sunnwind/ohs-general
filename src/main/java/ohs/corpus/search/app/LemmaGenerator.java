package ohs.corpus.search.app;

import java.util.Collection;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ohs.utils.Generics;

public class LemmaGenerator {

	private Properties prop = new Properties();

	private StanfordCoreNLP coreNLP;

	public LemmaGenerator() {
		prop = new Properties();
		prop.setProperty("annotators", "tokenize, ssplit, pos, lemma");

		coreNLP = new StanfordCoreNLP(prop);
	}

	public String getLemma(String word) throws Exception {
		Annotation anno = new Annotation(word);
		coreNLP.annotate(anno);

		List<CoreMap> sents = anno.get(SentencesAnnotation.class);

		StringBuffer sb = new StringBuffer();

		for (CoreMap sentence : sents) {
			// traversing the words in the current sentence
			// a CoreLabel is a CoreMap with additional token-specific methods

			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// String word = token.get(TextAnnotation.class);
				String pos = token.get(PartOfSpeechAnnotation.class);
				String lemma = token.get(LemmaAnnotation.class);
				sb.append(lemma + " ");
			}
		}

		word = sb.toString().trim();

		return word;
	}

	public List<String> getLemmas(Collection<String> words) throws Exception {
		List<String> ret = Generics.newArrayList(words.size());
		for (String word : words) {
			ret.add(getLemma(word));
		}
		return ret;
	}

}
