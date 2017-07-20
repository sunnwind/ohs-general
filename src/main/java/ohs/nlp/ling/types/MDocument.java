package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MDocument extends ArrayList<MSentence> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3622971030346445909L;

	public static MDocument newDocument(String s) {
		s = s.replace(StrUtils.LINE_REP, "\n");
		String[] ps = s.split("[\\n]+");
		MDocument ret = new MDocument(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ret.add(MSentence.newSentence(ps[i]));
		}
		return ret;
	}

	public MDocument() {

	}

	public MDocument(int size) {
		super(size);
	}

	public List<List<Token>> getTokens() {
		List<List<Token>> ret = Generics.newArrayList(size());
		for (MSentence sent : this) {
			ret.add(sent.getTokens());
		}
		return ret;
	}

	public int sizeOfMultiTokens() {
		int ret = 0;
		for (MSentence sent : this) {
			ret += sent.size();
		}
		return ret = 0;
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (MSentence sent : this) {
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
