package ohs.nlp.ling.types;

import java.util.ArrayList;
import java.util.List;

public class MCollection extends ArrayList<MDocument> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2528539276233604652L;

	public MCollection() {
		super();
	}

	public MCollection(int size) {
		super(size);
	}

	public MCollection(List<MDocument> m) {
		super(m);
	}

	public MCollection(MCollection m) {
		super(m);
	}

	public MCollection subCollection(int i, int j) {
		return new MCollection(subList(i, j));
	}

}
