package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

			MSentence sent = MSentence.newSentence(ps[i]);

			if (sent.size() > 0) {
				ret.add(sent);
			}
		}
		return ret;
	}

	private Map<String, String> attrMap = Generics.newHashMap();

	public MDocument() {

	}

	public MDocument(int size) {
		super(size);
	}

	public MDocument(List<MSentence> s) {
		super(s);
	}

	public Map<String, String> getAttrMap() {
		return attrMap;
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
		// sb.append("[attr map]");
		//
		// if (attrMap.size() > 0) {
		// for (String attr : Generics.newTreeSet(attrMap.keySet())) {
		// String val = attrMap.get(attr);
		// sb.append(String.format("\n%s:\t%s", attr, val));
		// }
		// }
		// sb.append("\n\n");

		for (int i = 0; i < size(); i++) {
			// sb.append(String.format("[sent-%d]", i + 1));
			sb.append(get(i).toString());
			if (i != size() - 1) {
				sb.append("\n\n");
			}
		}
		return sb.toString();
	}

}
