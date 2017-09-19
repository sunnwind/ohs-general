package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class Token extends ArrayList<String> {

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

	public static Token newToken(String s) {
		List<String> ps = StrUtils.split(s);
		Token ret = new Token(ps.size());
		for (int i = 0; i < ps.size(); i++) {
			ret.add(ps.get(i));
		}
		return ret;
	}

	protected int start = 0;

	public Token(int size) {
		super(size);
	}

	public Token(int start, String word) {
		this.start = start;
		add(word);
	}

	public String get(String attr) {
		String ret = null;
		int idx = INDEXER.indexOf(attr);
		if (idx != -1) {
			ret = get(idx);
		}
		return ret;
	}

	public int getStart() {
		return start;
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
	}

}
