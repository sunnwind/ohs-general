package ohs.ir.search.index;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import ohs.math.ArrayMath;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public abstract class InvertedIndex {

	public static IntegerMatrix findCollocations(IntegerArray poss_x, IntegerArray poss_y, boolean keep_order,
			int window_size) {
		List<IntegerArray> ret = Generics.newArrayList(Math.min(poss_x.size(), poss_y.size()));
		int i = 0;
		int j = 0;

		while (i < poss_x.size() && j < poss_y.size()) {
			int pos_x = poss_x.get(i);
			int pos_y = poss_y.get(j);
			int dist = pos_y - pos_x;

			if (!keep_order) {
				dist = Math.abs(dist);
			}

			if (dist > 0 && dist <= window_size) {
				IntegerArray p = new IntegerArray(2);
				p.add(pos_x);
				p.add(pos_y);
				ret.add(p);
			}

			if (pos_x == pos_y) {
				i++;
				j++;
			} else if (pos_x > pos_y) {
				j++;
			} else if (pos_x < pos_y) {
				i++;
			}
		}
		return new IntegerMatrix(ret);
	}

	public static PostingList findCollocations(PostingList pl_x, PostingList pl_y, boolean keep_order, int window_size)
			throws Exception {
		IntegerArray dseqs_x = pl_x.getDocSeqs();
		IntegerArray dseqs_y = pl_y.getDocSeqs();
		IntegerMatrix posData_x = pl_x.getPosData();
		IntegerMatrix posData_y = pl_y.getPosData();
		IntegerMatrix posData_x_min = pl_x.getEndPosData();

		int size = Math.min(pl_x.size(), pl_y.size());

		List<Integer> dseqs_xy = Generics.newArrayList(size);
		List<IntegerArray> posData_xy = Generics.newArrayList(size);
		List<IntegerArray> posData_xy_min = Generics.newArrayList(size);

		int i = 0;
		int j = 0;

		while (i < dseqs_x.size() && j < dseqs_y.size()) {
			int dseq_x = dseqs_x.get(i);
			int dseq_y = dseqs_y.get(j);

			if (dseq_x == dseq_y) {
				IntegerArray poss_x = posData_x.get(i);
				IntegerArray poss_y = posData_y.get(j);
				IntegerArray poss_x_min = posData_x_min == null ? new IntegerArray() : posData_x_min.get(i);

				IntegerMatrix ps = findCollocations(poss_x, poss_y, keep_order, window_size);

				if (ps.size() > 0) {
					Map<Integer, Integer> posToMin = Generics.newHashMap(poss_x.size());

					for (int k = 0; k < poss_x_min.size(); k++) {
						posToMin.put(poss_x.get(k), poss_x_min.get(k));
					}

					dseqs_xy.add(dseq_x);

					IntegerArray poss_xy = new IntegerArray(ps.size());
					IntegerArray poss_xy_min = new IntegerArray(ps.size()); // the
																			// minimum
																			// pos
																			// of
																			// word
																			// sequence

					for (IntegerArray p_xy : ps) {
						if (keep_order) {
							poss_xy.add(p_xy.get(1));
						} else {
							poss_xy.add(ArrayMath.max(p_xy.values()));
						}

						IntegerArray tmp = new IntegerArray(p_xy.size() + posToMin.size());
						tmp.addAll(p_xy);

						for (int pos : p_xy) {
							Integer min = posToMin.get(pos);

							if (min != null) {
								tmp.add(min);
							}
						}
						tmp.trimToSize();
						poss_xy_min.add(ArrayMath.min(tmp.values()));
					}

					posData_xy.add(poss_xy);
					posData_xy_min.add(poss_xy_min);
				}
				i++;
				j++;
			} else if (dseq_x > dseq_y) {
				j++;
			} else if (dseq_x < dseq_y) {
				i++;
			}
		}

		PostingList ret = new PostingList(pl_y.getWord(), new IntegerArray(dseqs_xy),
				new IntegerMatrix(posData_xy));

		IntegerMatrix m = new IntegerMatrix(posData_xy_min);
		ret.setEndPosData(m);
		return ret;
	}

	public static IntegerArray intersection(PostingList pl_x, PostingList pl_y) {
		IntegerArray dseqs_x = pl_x.getDocSeqs();
		IntegerArray dseqs_y = pl_y.getDocSeqs();
		List<Integer> dseqs_xy = Generics.newLinkedList();
		int i = 0;
		int j = 0;
		while (i < dseqs_x.size() && j < dseqs_y.size()) {
			int dseq1 = dseqs_x.get(i);
			int dseq2 = dseqs_y.get(j);

			if (dseq1 == dseq2) {
				dseqs_xy.add(dseq1);
				i++;
				j++;
			} else if (dseq1 > dseq2) {
				j++;
			} else if (dseq1 < dseq2) {
				i++;
			}
		}
		return new IntegerArray(dseqs_xy);
	}

	protected int doc_cnt = 0;

	protected boolean print_log = false;

	protected Vocab vocab;

	public void close() throws Exception {
	}

	abstract public InvertedIndex copyShallow() throws Exception;

	public int getDocCnt() {
		return doc_cnt;
	}

	abstract public PostingList getPostingList(int w) throws Exception;

	/**
	 * @param Q
	 * @param keep_order
	 * @param window_size
	 * @return
	 * @throws Exception
	 */
	public PostingList getPostingList(IntegerArray Q, boolean keep_order, int window_size) throws Exception {
		PostingList pl_x = null, pl_y = null;
		int i = 0;

		while (i < Q.size()) {
			int y = Q.get(i);
			pl_y = getPostingList(y);

			if (i > 0) {
				PostingList pl_xy = findCollocations(pl_x, pl_y, keep_order, window_size);
				pl_y = pl_xy;
			}

			if (pl_y.size() == 0) {
				break;
			}

			pl_x = pl_y;
			i++;
		}

		if (i == Q.size() && i > 1) {
			pl_y.setWord(-1);

			IntegerMatrix posData_min = pl_y.getEndPosData();
			IntegerMatrix posData = pl_y.getPosData();

			pl_y.setPosData(posData_min);
			pl_y.setEndPosData(posData);
		}

		return pl_y;
	}

	public PostingList getPostingListOld(IntegerArray Q, boolean keep_order, int window_size) throws Exception {
		PostingList pl_x = getPostingList(Q.get(0));
		PostingList ret = pl_x;

		if (pl_x != null && Q.size() > 1) {
			int i = 0;

			while (++i < Q.size()) {
				int y = Q.get(i);
				PostingList pl_y = getPostingList(y);

				if (pl_y == null) {
					break;
				}

				PostingList pl_xy = findCollocations(pl_x, pl_y, keep_order, window_size);

				if (pl_xy.size() == 0) {
					break;
				}

				pl_x = pl_xy;
			}

			if (i == Q.size()) {
				ret = pl_x;
				ret.setWord(-1);

				IntegerMatrix posData_min = ret.getEndPosData();
				IntegerMatrix posData = ret.getPosData();

				ret.setPosData(posData_min);
				ret.setEndPosData(posData);
			}
		}

		return ret;
	}

	public PostingList getPostingList(String word) throws Exception {
		PostingList ret = null;
		int w = vocab.indexOf(word);
		if (w != -1) {
			ret = getPostingList(w);
		}
		return ret;
	}

	public List<PostingList> getPostingLists(Collection<String> Q) throws Exception {
		List<PostingList> ret = Generics.newArrayList();
		for (String word : Q) {
			ret.add(getPostingList(word));
		}
		return ret;
	}

	abstract public List<PostingList> getPostingLists(int i, int j) throws Exception;

	public List<PostingList> getPostingLists(IntegerArray Q) throws Exception {
		List<PostingList> ret = Generics.newArrayList(Q.size());
		for (int w : Q) {
			ret.add(getPostingList(w));
		}
		return ret;
	}

	public void setPrintLog(boolean print_log) {
		this.print_log = print_log;
	}

	public void setVocab(Vocab vocab) {
		this.vocab = vocab;
	}

	abstract public int size();

}
