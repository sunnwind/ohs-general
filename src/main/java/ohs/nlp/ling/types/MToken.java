package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MToken extends ArrayList<Object> {

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

	public static MToken newToken(String s) {
		List<String> ps = StrUtils.split(s);
		MToken ret = new MToken(ps.size());
		for (int i = 0; i < ps.size(); i++) {
			ret.add(ps.get(i));
		}
		return ret;
	}

	protected int start = 0;

	public MToken() {
		super();
	}

	public MToken(int size) {
		super(size);
	}

	public MToken(int start, String word) {
		this.start = start;
		add(word);
	}

	public Object get(String attr) {
		Object ret = null;
		int idx = INDEXER.indexOf(attr);
		if (idx != -1) {
			ret = get(idx);
		}
		return ret;
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

		for (int i = 0; i < size; i++) {
			sb.append(get(i));
			if (i != size() - 1) {
				sb.append(DELIM);
			}
		}
		return sb.toString();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(start);
	}

}
