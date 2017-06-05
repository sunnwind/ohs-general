package ohs.corpus.type;

import java.util.List;

import ohs.nlp.ling.types.KDocument;
import ohs.nlp.ling.types.KSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.utils.Generics;

public class KoreanPosTokenizer extends StringTokenizer {

	public KoreanPosTokenizer() {
		super(new StringNormalizer());
	}

	public List<String> tokenize(String s) {
		List<String> ret = Generics.newArrayList();
		if (s.length() > 0) {
			KDocument doc = KDocument.newDocument(s);
			for (KSentence sent : doc) {
				for (MultiToken mts : sent) {
					for (Token t : mts) {
						String word = t.get(TokenAttr.WORD);
						String pos = t.get(TokenAttr.POS);
						word = word.toLowerCase();

						Token tt = new Token();
						tt.add(word);
						tt.add(pos);

						ret.add(tt.toString());
					}
				}
			}
		}
		return ret;
	}

}
