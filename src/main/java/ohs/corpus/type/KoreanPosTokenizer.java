package ohs.corpus.type;

import java.util.List;

import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.utils.Generics;

public class KoreanPosTokenizer extends StringTokenizer {

	public KoreanPosTokenizer() {
		super(new StringNormalizer());
	}

	public List<String> tokenize(String s) {
		List<String> ret = Generics.newArrayList();
		if (s.length() > 0) {
			LDocument doc = LDocument.newDocument(s);
			for (LSentence sent : doc) {
				for (LToken t : sent) {
					String word = t.getString(0);
					String pos = t.getString(1);
					word = word.toLowerCase();

					LToken tt = new LToken();
					tt.add(word);
					tt.add(pos);

//					ret.add(tt.toString().replace(" / ", "_/_"));
				}
			}
		}
		return ret;
	}

}
