package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.matrix.DenseVector;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class LToken extends ArrayList<Object> {

	public static String DELIM = " ";

	public static Indexer<String> INDEXER = Generics.newIndexer();

	/**
	 * 
	 */
	private static final long serialVersionUID = 7950967445551151259L;

	static {
		INDEXER = Generics.newIndexer(2);
		INDEXER.add("WORD");
		INDEXER.add("POS");
	}

	public static LToken newToken(String s) {
		List<String> ps = StrUtils.split(s);
		LToken ret = new LToken(ps.size());
		for (int i = 0; i < ps.size(); i++) {
			ret.add(ps.get(i));
		}
		return ret;
	}

	private DenseVector fv = null;

	protected int start = 0;

	public LToken() {
		super();
	}

	public LToken(int size) {
		super(size);
	}

	public LToken(int start, String word) {
		this.start = start;
		add(word);
	}

	public LToken(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public LToken(String word) {
		this(0, word);
	}

	public Object get(String attr) {
		Object ret = null;
		int idx = INDEXER.indexOf(attr);
		if (idx != -1) {
			ret = get(idx);
		}
		return ret;
	}

	public DenseVector getFeatureVector() {
		return fv;
	}

	public int getStart() {
		return start;
	}

	public String getString(int i) {
		return get(i).toString();
	}

	public List<String> getStrings(int i, int j) {
		List<String> ret = Generics.newArrayList(j - i);
		for (int k = i; k < j; k++) {
			ret.add(getString(k));
		}
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		start = ois.readInt();
		int size = ois.readInt();
		for (int i = 0; i < size; i++) {
			add(ois.readObject());
		}

		if (ois.readBoolean()) {
			fv = new DenseVector(ois);
		}
	}

	public void setFeatureVector(DenseVector fv) {
		this.fv = fv;
	}

	public void setStart(int start) {
		this.start = start;
	}

	@Override
	public String toString() {
		return toString(true);
	}

	public String toString(boolean print_attr_names) {
		StringBuffer sb = new StringBuffer();

		int size = INDEXER.size();

		if (size == 0) {
			size = size();
		}

		for (int i = 0; i < size(); i++) {
			sb.append(get(i));
			if (i != size() - 1) {
				sb.append(DELIM);
			}
		}
		return sb.toString();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(start);
		oos.writeInt(size());
		for (Object o : this) {
			oos.writeObject(o);
		}

		if (fv == null) {
			oos.writeBoolean(false);
		} else {
			oos.writeBoolean(true);
			fv.writeObject(oos);
		}
	}

}
