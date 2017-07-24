package ohs.corpus.type;

import java.util.List;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
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
			MDocument doc = MDocument.newDocument(s);
			for (MSentence sent : doc) {
				for (MultiToken mts : sent) {
					for (Token t : mts) {
						String word = t.get(0);
						String pos = t.get(1);
						word = word.toLowerCase();

						Token tt = new Token();
						tt.add(word);
						tt.add(pos);

						ret.add(tt.toString().replace(" / ", "_/_"));
					}
				}
			}
		}
		return ret;
	}

}
