package ohs.nlp.ling.types;

import java.util.ArrayList;

public class MultiToken extends ArrayList<Token> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1548339270826089537L;

	public static final String DELIM = " + ";

	public static MultiToken newMultiToken(String s) {
		String[] ps = s.split(" \\+ ");
		MultiToken ret = new MultiToken(ps.length);

		for (int i = 0; i < ps.length; i++) {
			ret.add(Token.newToken(ps[i]));
		}

		return ret;
	}

	private int start = 0;

	public MultiToken() {

	}

	public MultiToken(int size) {
		super(size);
	}

	public MultiToken(int start, String word) {
		this.start = start;
		add(Token.newToken(word));
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(get(i).toString());
			if (i != size() - 1) {
				sb.append(DELIM);
			}
		}
		return sb.toString();
	}

}
