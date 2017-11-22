package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.queryparser.flexible.standard.parser.Token;

import ohs.matrix.DenseVector;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MSentence extends ArrayList<MToken> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3571196248876992657L;

	public static final String START = "<s>";

	public static final String END = "</s>";

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
				String str = ps.get(j).trim();
				if (str.length() == 0) {
					continue;
				}
				t.add(str);
			}

			if (num_attrs == t.size()) {
				ret.add(t);
			} else {
				System.err.printf("missing values found:\t%s\n", ps.toString());
				System.exit(0);
			}
		}
		return ret;
	}

	public boolean is_padded = false;

	private Map<String, String> attrMap = Generics.newHashMap();

	private DenseVector fv;

	public MSentence() {

	}

	public MSentence toPaddedSentence() {
		MSentence ret = new MSentence(size() + 2);
		MToken t1 = new MToken();
		MToken t2 = new MToken();

		t1.add(START);
		t2.add(START);

		for (int j = 1; j < MToken.INDEXER.size(); j++) {
			t1.add("");
			t2.add("");
		}

		ret.add(t1);
		ret.addAll(this);
		ret.add(t2);
		ret.setIsPadded(true);
		return ret;
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

	public DenseVector getFeatureVector() {
		return fv;
	}

	public List<String> getTokenStrings(int idx) {
		List<String> ret = Generics.newArrayList(size());
		for (MToken t : this) {
			ret.add(t.getString(idx));
		}
		return ret;
	}

	public boolean isPadded() {
		return is_padded;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
	}

	public void setFeatureVector(DenseVector fv) {
		this.fv = fv;
	}

	public void setIsPadded(boolean is_padded) {
		this.is_padded = is_padded;
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
