package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MSentence extends ArrayList<MToken> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3571196248876992657L;

	public static MSentence newSentence(String s) {
		s = s.replace("<nl>", "\n");
		String[] lines = s.split("\n");
		MSentence ret = new MSentence(lines.length);

		int num_attrs = MToken.INDEXER.size();

		for (int i = 0; i < lines.length; i++) {
			List<String> ps = StrUtils.split(lines[i]);

			int min_attrs = Math.min(num_attrs, ps.size());
			MToken t = new MToken(min_attrs);

			for (int j = 0; j < min_attrs; j++) {
				t.add(ps.get(j));
			}

			if (num_attrs == t.size()) {
				ret.add(t);
			} else {
				System.err.printf("missing values found:\t%s\n", ps.toString());
			}
		}
		return ret;
	}

	private Map<String, String> attrMap = Generics.newHashMap();

	public MSentence() {

	}

	public MSentence(int size) {
		super(size);
	}

	public MSentence(List<MToken> ret) {
		super(ret);
	}

	public Map<String, String> getAttrMap() {
		return attrMap;
	}

	public List<String> getTokenStrings(int idx) {
		List<String> ret = Generics.newArrayList(size());
		for (MToken t : this) {
			ret.add(t.getString(idx));
		}
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
	}

	public MSentence subSentence(int i, int j) {
		return new MSentence(subList(i, j));
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

	public void writeObject(ObjectOutputStream oos) throws Exception {
	}

}
