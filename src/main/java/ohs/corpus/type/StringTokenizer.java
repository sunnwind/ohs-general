package ohs.corpus.type;

import java.util.List;

import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class StringTokenizer {

	protected StringNormalizer sn;

	public StringTokenizer() {
		this(null);
	}

	public StringTokenizer(StringNormalizer sn) {
		this.sn = sn;
	}

	public List<List<String>> tokenize(List<String> s) {
		List<List<String>> ret = Generics.newArrayList(s.size());
		for (int i = 0; i < s.size(); i++) {
			ret.add(tokenize(s.get(i)));
		}
		return ret;
	}

	public StringNormalizer getStringNormalizer() {
		return sn;
	}

	public List<String> tokenize(String s) {
		List<String> ret = StrUtils.split(s);
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
