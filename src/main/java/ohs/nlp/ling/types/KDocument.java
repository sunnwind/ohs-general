package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;

public class KDocument extends ArrayList<KSentence> {

	public static KDocument newDocument(String s) {
		String[] ps = s.split("[\\n]+");
		KDocument ret = new KDocument(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ret.add(KSentence.newSentence(ps[i]));
		}
		return ret;
	}

	public KDocument() {

	}

	public KDocument(int size) {
		super(size);
	}

	public List<List<Token>> getTokens() {
		List<List<Token>> ret = Generics.newArrayList(size());
		for (KSentence sent : this) {
			ret.add(sent.getTokens());
		}
		return ret;
	}

	public int sizeOfMultiTokens() {
		int ret = 0;
		for (KSentence sent : this) {
			ret += sent.size();
		}
		return ret = 0;
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (KSentence sent : this) {
			ret += sent.sizeOfTokens();
		}
		return ret = 0;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(get(i).toString());
			if (i != size() - 1) {
				sb.append("\n");
			}
		}
		return sb.toString();
	}

}
