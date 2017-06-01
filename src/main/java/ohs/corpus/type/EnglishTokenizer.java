package ohs.corpus.type;

import java.util.List;

import ohs.ir.medical.general.NLPUtils;

public class EnglishTokenizer extends StringTokenizer {

	public EnglishTokenizer() {
		this(new EnglishNormalizer());
	}

	public EnglishTokenizer(EnglishNormalizer en) {
		super(en);
	}

	public List<String> tokenize(String s) {
		List<String> ret = NLPUtils.tokenize(s);
		for (int i = 0; i < ret.size(); i++) {
			String tok = ret.get(i);
			if (sn != null) {
				tok = sn.normalize(tok);
			}
			ret.set(i, tok);
		}
		return ret;
	}

}
