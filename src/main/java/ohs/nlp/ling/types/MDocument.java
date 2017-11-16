package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ohs.matrix.DenseVector;
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

	private DenseVector fv;

	public MDocument() {

	}

	public MDocument toPaddedDocument() {
		MDocument ret = new MDocument(size());
		for (MSentence s : this) {
			ret.add(s.toPaddedSentence());
		}
		return ret;
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

	public DenseVector getFeatureVector() {
		return fv;
	}

	public MSentence getTokens() {
		List<MToken> ret = Generics.newArrayList(sizeOfTokens());
		for (MSentence s : this) {
			ret.addAll(s);
		}
		return new MSentence(ret);
	}

	public List<String> getTokenStrings(int idx) {
		List<String> ret = Generics.newArrayList(sizeOfTokens());
		for (MSentence s : this) {
			ret.addAll(s.getTokenStrings(idx));
		}
		return ret;
	}

	public void setFeatureVector(DenseVector fv) {
		this.fv = fv;
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (MSentence s : this) {
			ret += s.size();
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
