package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;

public class MDocument extends ArrayList<MSentence> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3622971030346445909L;

	public static MDocument newDocument(String s) {
		s = s.replace("<nl>", "\n");
		String[] ps = s.split("\n\n");
		MDocument ret = new MDocument(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ps[i] = ps[i].trim();
			if (ps[i].length() == 0) {
				continue;
			}
			ret.add(MSentence.newSentence(ps[i]));
		}
		return ret;
	}

	public MDocument() {

	}

	public MDocument(int size) {
		super(size);
	}

	public MDocument(List<MSentence> s) {
		super(s);
	}

	public MSentence getTokens() {
		List<MToken> ret = Generics.newArrayList(sizeOfTokens());
		for (MSentence s : this) {
			ret.addAll(s);
		}
		return new MSentence(ret);
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (MSentence sent : this) {
			ret += sent.size();
		}
		return ret;
	}

	public MDocument subDocument(int i, int j) {
		return new MDocument(subList(i, j));
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(get(i).toString());
			if (i != size() - 1) {
				sb.append("\n\n");
			}
		}
		return sb.toString();
	}

}
