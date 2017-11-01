package ohs.corpus.type;

import java.util.List;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
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
				for (MToken t : sent) {
					String word = t.getString(0);
					String pos = t.getString(1);
					word = word.toLowerCase();

					MToken tt = new MToken();
					tt.add(word);
					tt.add(pos);

//					ret.add(tt.toString().replace(" / ", "_/_"));
				}
			}
		}
		return ret;
	}

}
