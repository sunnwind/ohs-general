package ohs.nlp.pos;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.TokenAttr;
import ohs.utils.StrUtils;

public class SejongParser {

	public static MultiToken parseMultiToken(String s) {
		String[] two = s.split("\t");

		String surface = two[0];
		String[] parts = two[1].split(MultiToken.DELIM.replace("+", "\\+"));
		MToken[] toks = new MToken[parts.length];
		for (int i = 0; i < parts.length; i++) {
			toks[i] = parseToken(parts[i]);
		}
		MultiToken ret = new MultiToken(0, surface);
		ret.setSubTokens(toks);

		return ret;
	}

	public static MDocument parseDocument(String s) {
		String[] lines = s.split("\n\n");
		MSentence[] sents = new MSentence[lines.length];
		for (int i = 0; i < sents.length; i++) {
			sents[i] = parseSentence(lines[i]);
		}

		MDocument doc = new MDocument(sents);

		enumerateStarts(doc);

		return doc;
	}

	public static void enumerateStarts(MDocument doc) {
		MToken[] mts = doc.toMultiTokens();
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

	public static MSentence parseSentence(String s) {
		String[] lines = s.split("\n");
		MultiToken[] mts = new MultiToken[lines.length];
		for (int i = 0; i < lines.length; i++) {
			mts[i] = parseMultiToken(lines[i]);
		}
		return new MSentence(mts);
	}

	public static MToken parseToken(String s) {
		String[] values = new String[TokenAttr.size()];

		StrUtils.copy(s.split(MToken.DELIM), values);

		MToken ret = new MToken();
		for (TokenAttr attr : TokenAttr.values()) {
			ret.set(attr, values[attr.ordinal()]);
		}
		return ret;
	}

}
