package ohs.corpus.type;

import ohs.utils.StrUtils;

public class StringNormalizer {

	public String normalize(String s) {
		s = StrUtils.normalizeNumbers(s);
		s = StrUtils.normalizeSpaces(s);
		return s.toLowerCase();
	}

}
