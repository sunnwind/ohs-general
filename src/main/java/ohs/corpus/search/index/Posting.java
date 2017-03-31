package ohs.corpus.search.index;

import ohs.types.number.IntegerArray;

public class Posting {
	private int dseq;

	private IntegerArray poss;

	private IntegerArray ends;

	public Posting(int dseq, IntegerArray poss) {
		this(dseq, poss, null);
	}

	public Posting(int dseq, IntegerArray poss, IntegerArray ends) {
		super();
		this.dseq = dseq;
		this.poss = poss;
		this.ends = ends;
	}

	public int getDocseq() {
		return dseq;
	}

	public IntegerArray getEnds() {
		return ends;
	}

	public IntegerArray getPoss() {
		return poss;
	}

	public int size() {
		return poss.size();
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("dseq=[%d]", dseq));
		sb.append(", wlocs=[");
		for (int i = 0; i < poss.size(); i++) {
			sb.append(poss.get(i));
			if (i != poss.size() - 1) {
				sb.append(",");
			}
		}
		sb.append("]");
		return sb.toString();
	}

}