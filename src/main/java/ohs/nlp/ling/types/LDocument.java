package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.matrix.Vector;
import ohs.types.generic.Counter;
import ohs.utils.Generics;

public class LDocument extends ArrayList<LSentence> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3622971030346445909L;

	public static LDocument newDocument(String s) {
		s = s.replace("<nl>", "\n");
		String[] ps = s.split("\n\n");
		LDocument ret = new LDocument(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ps[i] = ps[i].trim();

			if (ps[i].length() == 0) {
				continue;
			}

			LSentence sent = LSentence.newSentence(ps[i]);

			if (sent.size() > 0) {
				ret.add(sent);
			}
		}
		return ret;
	}

	private Map<String, String> attrMap = Generics.newHashMap();

	private Vector fv;

	public LDocument() {

	}

	public LDocument(int size) {
		super(size);
	}

	public LDocument(List<LSentence> s) {
		super(s);
	}

	public LDocument(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public LDocument clone() {
		LDocument ret = new LDocument(size());
		for (LSentence s : this) {
			ret.add(s.clone());
		}
		HashMap<String, String> am = (HashMap<String, String>) attrMap;
		ret.setAttrMap((Map<String, String>) am.clone());
		ret.setFeatureVector(fv == null ? null : fv.copy());
		return ret;
	}

	public void doPadding() {
		for (LSentence s : this) {
			s.doPadding();
		}
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

	public LSentence getTokens() {
		List<LToken> ret = Generics.newArrayList(sizeOfTokens());
		for (LSentence s : this) {
			ret.addAll(s);
		}
		return new LSentence(ret);
	}

	public List<String> getTokenStrings(int idx) {
		return getTokens().getTokenStrings(idx);
	}

	public List<String> getTokenStrings(int[] idxs, String glue) {
		return getTokens().getTokenStrings(idxs, glue);
	}

	public void readObject(ObjectInputStream ois) throws Exception {

		{
			int size = ois.readInt();
			for (int i = 0; i < size; i++) {
				LSentence t = new LSentence();
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

	public int sizeOfTokens() {
		int ret = 0;
		for (LSentence s : this) {
			ret += s.size();
		}
		return ret;
	}

	public LDocument subDocument(int i, int j) {
		return new LDocument(subList(i, j));
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

	public void writeObject(ObjectOutputStream oos) throws Exception {
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
