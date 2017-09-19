package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class MSentence extends ArrayList<MultiToken> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3571196248876992657L;

	public static MSentence newSentence(String s) {
		s = s.replace("<nl>", "\n");
		String[] lines = s.split("\n");
		MSentence ret = new MSentence(lines.length);

		int num_attrs = Token.INDEXER.size();

		for (int i = 0; i < lines.length; i++) {
			List<String> ps = StrUtils.split(lines[i]);

			int min_attrs = Math.min(num_attrs, ps.size());
			Token t = new Token(min_attrs);

			for (int j = 0; j < min_attrs; j++) {
				t.add(ps.get(j));
			}

			if (num_attrs == t.size()) {
				MultiToken mt = new MultiToken();
				mt.add(t);
				ret.add(mt);
			} else {
				System.err.printf("missing values found:\t%s\n", ps.toString());
			}
		}
		return ret;
	}

	private boolean is_multi_tok = false;

	public MSentence() {

	}

	public MSentence(int size) {
		super(size);
	}

	public List<Token> getTokens() {
		List<Token> ret = Generics.newArrayList(sizeOfTokens());
		for (MultiToken mt : this) {
			for (Token t : mt) {
				ret.add(t);
			}
		}
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		is_multi_tok = ois.readBoolean();
	}

	public int sizeOfTokens() {
		int ret = 0;
		for (MultiToken mt : this) {
			ret += mt.size();
		}
		return ret = 0;
	}

	public MSentence subSentence(int i, int j) {
		MSentence ret = new MSentence(j - i);
		for (MultiToken mt : subList(i, j)) {
			ret.add(mt);
		}
		return ret;
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
		oos.writeBoolean(is_multi_tok);
	}

}
