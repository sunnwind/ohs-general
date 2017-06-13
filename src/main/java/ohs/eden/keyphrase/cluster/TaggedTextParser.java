package ohs.eden.keyphrase.cluster;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;

public class TaggedTextParser {

	public static MDocument parse(String s) {
		String[] lines = s.split("\\\\n");
		MSentence[] sents = new MSentence[lines.length];

		int num_mts = 0;
		int num_ts = 0;

		for (int i = 0; i < lines.length; i++) {
			String[] parts = lines[i].split("\\\\t");
			MultiToken[] toks = new MultiToken[parts.length];

			for (int j = 0; j < parts.length; j++) {
				String part = parts[j];
				String[] subParts = part.split(MultiToken.DELIM.replace("+", "\\+"));
				Token[] subToks = new Token[subParts.length];

				for (int k = 0; k < subParts.length; k++) {
					String subPart = subParts[k];
					String[] two = subPart.split(Token.DELIM);

					Token tok = new Token();

					if (two.length == 2) {
						String word = two[0].trim();
						String pos = two[1].trim();

						tok.set(TokenAttr.WORD, word);
						tok.set(TokenAttr.POS, pos);
					} else {
						tok.set(TokenAttr.WORD, "");
						tok.set(TokenAttr.POS, "");
					}

					tok.setStart(num_ts++);
					subToks[k] = tok;
				}

				MultiToken mt = new MultiToken();
				mt.setSubTokens(subToks);
				mt.setStart(num_mts++);
				toks[j] = mt;

			}
			sents[i] = new MSentence(toks);
		}

		MDocument doc = new MDocument(sents);
		return doc;
	}

}
