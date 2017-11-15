package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;

public class MDocumentCollection extends ArrayList<MDocument> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8530460881325620209L;

	public MDocument getSentences() {
		int size = 0;
		for (MDocument d : this) {
			size += d.size();
		}
		List<MSentence> ret = Generics.newArrayList(size);

		for (MDocument d : this) {
			for (MSentence s : d) {
				ret.add(s);
			}
		}
		return new MDocument(ret);
	}

	public MSentence getTokens() {
		List<MToken> ret = Generics.newArrayList(sizeOfTokens());
		for (MDocument d : this) {
			ret.addAll(d.getTokens());
		}
		return new MSentence(ret);
	}

	public List<String> getTokenStrings(int idx) {
		List<String> ret = Generics.newArrayList(sizeOfTokens());
		for (MDocument d : this) {
			ret.addAll(d.getTokenStrings(idx));
		}
		return ret;
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (MDocument d : this) {
			ret += d.sizeOfTokens();
		}
		return ret;
	}

}
