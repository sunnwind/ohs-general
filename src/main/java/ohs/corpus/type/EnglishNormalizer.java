package ohs.corpus.type;

import org.tartarus.snowball.ext.PorterStemmer;

import ohs.utils.StrUtils;

public class EnglishNormalizer extends StringNormalizer {

	private boolean stem = false;

	private PorterStemmer stemmer = new PorterStemmer();

	@Override
	public String normalize(String s) {
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
