package ohs.nlp.ling.types;

public enum TokenAttr {
	KWD, POS, WORD;
	// WORD, STEM, LEMMA, POS, NER

	public static int size() {
		return TokenAttr.values().length;
	}

	public static String[] strValues() {
		String[] ret = new String[values().length];
		for (TokenAttr attr : values()) {
			ret[attr.ordinal()] = attr.toString();
		}
		return ret;
	}
}
