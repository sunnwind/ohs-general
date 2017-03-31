package ohs.types.common;

import java.io.ObjectInputStream;
import java.io.ObjectOutput;

import ohs.types.generic.Pair;

public class StrPair extends Pair<String, String> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7125783010793193515L;

	public StrPair(String first, String second) {
		super(first, second);
	}

	public StrPair(String[] s) {
		this(s[0], s[1]);
	}

	public StrPair() {
		super("", "");
	}

	public void read(ObjectInputStream ois) throws Exception {
		first = ois.readUTF();
		second = ois.readUTF();
	}

	public void write(ObjectOutput oos) throws Exception {
		oos.writeUTF(first);
		oos.writeUTF(second);
	}

	public String[] asArray() {
		return new String[] { first, second };
	}

}
