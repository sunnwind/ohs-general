package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class Token extends ArrayList<String> {

	public static final String DELIM = " / ";
	/**
	 * 
	 */
	private static final long serialVersionUID = 7950967445551151259L;

	public static Token newToken(String s) {
		String[] ps = s.split(DELIM);
		Token ret = new Token(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ret.add(ps[i]);
		}
		return ret;
	}

	protected int start = 0;

	public Token() {

	}

	public Token(int size) {
		super(size);
	}

	public Token(int start, String word) {
		this();
		this.start = start;
		add(word);
	}

	public String get(TokenAttr attr) {
		return get(attr.ordinal());
	}

	public String[] get(TokenAttr[] attrs) {
		String[] ret = new String[attrs.length];
		for (int i = 0; i < attrs.length; i++) {
			ret[i] = get(attrs[i].ordinal());
		}
		return ret;
	}

	public int getStart() {
		return start;
	}

	public int length() {
		return get(TokenAttr.WORD).length();
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
		// if (print_attr_names) {
		// for (int i = 0; i < size(); i++) {
		// sb.append(TokenAttr.values()[i]);
		// if (i != size()) {
		// sb.append("\t");
		// }
		// }
		// sb.append("\n");
		// }
		// sb.append(StrUtils.join("\t", StrUtils.wrap(this)));

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
