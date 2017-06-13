package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.utils.Generics;

public class MSentence extends ArrayList<MultiToken> {

	public static final String DELIM_SENT = " __ ";

	public static MSentence newSentence(String s) {
		String[] ps = s.split(" __ ");
		MSentence ret = new MSentence(ps.length);
		for (int i = 0; i < ps.length; i++) {
			ret.add(MultiToken.newMultiToken(ps[i]));
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

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(get(i).toString());
			if (i != size() - 1) {
				sb.append(DELIM_SENT);
			}
		}
		return sb.toString();

	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeBoolean(is_multi_tok);
	}

}
