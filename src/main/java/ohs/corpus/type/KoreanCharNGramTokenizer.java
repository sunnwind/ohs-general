package ohs.corpus.type;

import java.util.List;

import ohs.utils.Generics;

public class KoreanCharNGramTokenizer extends StringTokenizer {

	protected int gram_size = 3;

	public KoreanCharNGramTokenizer() {
		super(new StringNormalizer());
	}

	public List<String> tokenize(String s) {
		List<String> ret = Generics.newArrayList();
		for (int i = 0; i < s.length() - gram_size + 1; i++) {
			int j = i + gram_size;
			String gram = s.substring(i, j);
			if (sn != null) {
				gram = sn.normalize(gram);
			}
			ret.add(gram);
		}
		return ret;
	}

}
