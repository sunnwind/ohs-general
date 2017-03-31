package ohs.corpus.type;

import org.tartarus.snowball.ext.PorterStemmer;

import ohs.ir.medical.general.NLPUtils;
import ohs.utils.StrUtils;

public class SimpleStringNormalizer implements StringNormalizer {

	private boolean tokenize = false;

	private boolean stem = false;

	private PorterStemmer stemmer = new PorterStemmer();

	public SimpleStringNormalizer(boolean tokenize) {
		this.tokenize = tokenize;
	}

	@Override
	public String normalize(String s) {
		if (tokenize) {
			s = StrUtils.join(" ", NLPUtils.tokenize(s));
		}
		s = StrUtils.normalizeNumbers(s);
		s = StrUtils.normalizeSpaces(s);
		if (stem) {
			s = stem(s);
		}
		return s.toLowerCase();
	}

	public void setStem(boolean stem) {
		this.stem = stem;

	}

	public void setTokenize(boolean tokenzie) {
		this.tokenize = tokenzie;
	}

	public String stem(String s) {
		String[] words = s.split(" ");
		for (int i = 0; i < words.length; i++) {
			stemmer.setCurrent(words[i]);
			if (stemmer.stem()) {
				words[i] = stemmer.getCurrent();
			}
		}
		return StrUtils.join(" ", words);
	}

}
