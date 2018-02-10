package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.crypto.spec.DESedeKeySpec;

import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.matrix.Vector;
import ohs.types.generic.Counter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class LSentence extends ArrayList<LToken> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3571196248876992657L;

	public static final String STARTING = "<bos>";

	public static final String ENDING = "<eos>";

	public static final String PADDING = "PADDING";

	public static LSentence newSentence(String s) {
		s = s.replace("<nl>", "\n");
		String[] lines = s.split("\n");
		LSentence ret = new LSentence(lines.length);

		int num_attrs = LToken.INDEXER.size();

		for (int i = 0; i < lines.length; i++) {
			List<String> ps = StrUtils.split(lines[i]);

			int min_attrs = Math.min(num_attrs, ps.size());
			LToken t = new LToken(min_attrs);

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

	private Vector fv;

	public LSentence() {

	}

	public LSentence(int size) {
		super(size);
	}

	public LSentence(List<LToken> ret) {
		super(ret);
	}

	public LSentence(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public LSentence clone() {
		LSentence ret = new LSentence(size());

		for (LToken t : this) {
			ret.add(t.clone());
		}

		ret.setAttrMap((Map<String, String>) ((HashMap<String, String>) attrMap).clone());
		ret.setFeatureVector(fv == null ? null : fv.copy());
		return ret;
	}

	public void doPadding() {
		LToken t1 = new LToken();
		LToken t2 = new LToken();

		t1.add(STARTING);
		t2.add(ENDING);

		for (int j = 1; j < LToken.INDEXER.size(); j++) {
			t1.add(PADDING);
			t2.add(PADDING);
		}

		t1.setIsPadding(true);
		t2.setIsPadding(true);

		List<LToken> ts = Generics.newArrayList(size() + 2);
		ts.add(t1);
		ts.addAll(this);
		ts.add(t2);

		is_padded = true;

		clear();

		addAll(ts);
	}

	public Map<String, String> getAttrMap() {
		return attrMap;
	}

	public Counter<String> getCounter(int idx) {
		Counter<String> ret = Generics.newCounter();
		for (String t : getTokenStrings(idx)) {
			ret.incrementCount(t, 1);
		}
		return ret;
	}

	public Vector getFeatureVector() {
		return fv;
	}

	public List<String> getTokenStrings(int idx) {
		List<String> ret = Generics.newArrayList(size());
		for (LToken t : this) {
			ret.add(t.getString(idx));
		}
		return ret;
	}

	public List<String> getTokenStrings(int[] idxs, String glue) {
		List<String> ret = Generics.newArrayList(size());
		for (LToken t : this) {
			List<String> l = Generics.newArrayList(idxs.length);
			for (int idx : idxs) {
				l.add(t.getString(idx));
			}
			ret.add(StrUtils.join(glue, l));
		}
		return ret;
	}

	public boolean isPadded() {
		return is_padded;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		is_padded = ois.readBoolean();

		{
			int size = ois.readInt();
			for (int i = 0; i < size; i++) {
				LToken t = new LToken();
				t.readObject(ois);
				add(t);
			}
		}

		{
			int size = ois.readInt();

			for (int i = 0; i < size; i++) {
				attrMap.put(ois.readUTF(), ois.readUTF());
			}
		}

		{
			short type = ois.readShort();

			if (type == 1) {
				fv = new DenseVector(ois);
			} else if (type == 2) {
				fv = new SparseVector(ois);
			}
		}
	}

	public void setAttrMap(Map<String, String> attrMap) {
		this.attrMap = attrMap;
	}

	public void setFeatureVector(Vector fv) {
		this.fv = fv;
	}

	public void setIsPadded(boolean is_padded) {
		this.is_padded = is_padded;
	}

	public LSentence subSentence(int i, int j) {
		return new LSentence(subList(i, j));
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

	public LDocument toDocument() {
		LDocument ret = new LDocument(1);
		ret.add(this);
		return ret;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeBoolean(is_padded);

		{
			oos.writeInt(size());
			for (int i = 0; i < size(); i++) {
				get(i).writeObject(oos);
			}
		}

		{
			oos.writeInt(attrMap.size());
			for (Entry<String, String> e : attrMap.entrySet()) {
				oos.writeUTF(e.getKey());
				oos.writeUTF(e.getValue());
			}
		}

		{
			if (fv == null) {
				oos.writeShort(0);
			} else {
				if (fv instanceof DenseVector) {
					oos.writeShort(1);
					DenseVector v = (DenseVector) fv;
					v.writeObject(oos);
				} else {
					oos.writeShort(2);
					SparseVector v = (SparseVector) fv;
					v.writeObject(oos);
				}
			}
		}
	}

}
