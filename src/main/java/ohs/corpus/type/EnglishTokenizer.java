package ohs.corpus.type;

import java.util.List;

import ohs.ir.medical.general.NLPUtils;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class EnglishTokenizer extends StringTokenizer {

	private boolean use_lexical_tokenizer = true;

	public EnglishTokenizer() {
		this(new EnglishNormalizer(), true);
	}

	public EnglishTokenizer(EnglishNormalizer en, boolean use_lexical_tokenizer) {
		super(en);
		this.use_lexical_tokenizer = use_lexical_tokenizer;
	}

	public List<String> tokenize(String s) {
		List<String> ret = Generics.newArrayList();

		if (use_lexical_tokenizer) {
			s = StrUtils.join(" ", NLPUtils.tokenize(s));
		}

		ret = StrUtils.split(s);

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
