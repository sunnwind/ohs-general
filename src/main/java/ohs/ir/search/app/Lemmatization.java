package ohs.ir.search.app;

import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;

public class Lemmatization {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		List<String> phrss = Generics.newLinkedList();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_filtered.txt")) {
			String[] ps = line.split("\t");
			phrss.add(ps[0]);
		}

		Lemmatization l = new Lemmatization();
		l.lemmatize(phrss, MIRPath.DATA_DIR + "phrs/lemma.txt");

		System.out.println("process ends.");
	}

	public Lemmatization() {

	}

	public void lemmatize(List<String> phrss, String outFileName) throws Exception {

		Properties prop = new Properties();
		prop.setProperty("annotators", "tokenize, ssplit, pos, lemma");

		StanfordCoreNLP coreNLP = new StanfordCoreNLP(prop);

		Map<String, String> m = Generics.newHashMap();

		for (String phrs : phrss) {

			Annotation anno = new Annotation(phrs);

			coreNLP.annotate(anno);

			List<CoreMap> sents = anno.get(SentencesAnnotation.class);

			for (CoreMap sentence : sents) {
				// traversing the words in the current sentence
				// a CoreLabel is a CoreMap with additional token-specific
				// methods

				for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
					String word = token.get(TextAnnotation.class);
					String pos = token.get(PartOfSpeechAnnotation.class);
					String lemma = token.get(LemmaAnnotation.class);

					// if (!lemma.equals(word)) {
					m.put(word, lemma);
					// }
				}
			}
		}

		FileUtils.writeStringMapAsText(outFileName, m);
	}

}
