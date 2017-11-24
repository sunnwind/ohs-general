package ohs.nlp.pos;

import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.TokenAttr;
import ohs.utils.StrUtils;

public class SejongParser {

	public static MultiToken parseMultiToken(String s) {
		String[] two = s.split("\t");

		String surface = two[0];
		String[] parts = two[1].split(MultiToken.DELIM.replace("+", "\\+"));
		LToken[] toks = new LToken[parts.length];
		for (int i = 0; i < parts.length; i++) {
			toks[i] = parseToken(parts[i]);
		}
		MultiToken ret = new MultiToken(0, surface);
		ret.setSubTokens(toks);

		return ret;
	}

	public static LDocument parseDocument(String s) {
		String[] lines = s.split("\n\n");
		LSentence[] sents = new LSentence[lines.length];
		for (int i = 0; i < sents.length; i++) {
			sents[i] = parseSentence(lines[i]);
		}

		LDocument doc = new LDocument(sents);

		enumerateStarts(doc);

		return doc;
	}

	public static void enumerateStarts(LDocument doc) {
		LToken[] mts = doc.toMultiTokens();
		for (int i = 0, loc = 0; i < mts.length; i++) {
			MultiToken mt = (MultiToken) mts[i];
			mt.setStart(loc);

			for (int j = 0; j < mt.size(); j++) {
				mt.getToken(j).setStart(j);
			}

			loc += mt.length();
			loc++;
		}
	}

	public static LSentence parseSentence(String s) {
		String[] lines = s.split("\n");
		MultiToken[] mts = new MultiToken[lines.length];
		for (int i = 0; i < lines.length; i++) {
			mts[i] = parseMultiToken(lines[i]);
		}
		return new LSentence(mts);
	}

	public static LToken parseToken(String s) {
		String[] values = new String[TokenAttr.size()];

		StrUtils.copy(s.split(LToken.DELIM), values);

		LToken ret = new LToken();
		for (TokenAttr attr : TokenAttr.values()) {
			ret.set(attr, values[attr.ordinal()]);
		}
		return ret;
	}

}
