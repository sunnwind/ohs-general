package ohs.corpus.search.index;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;

public class InvertedIndex {

	public static final String DATA_NAME = "posting.ser";

	public static final String META_NAME = "posting_meta.ser";

	public static IntegerArrayMatrix findCollocations(IntegerArray poss_x, IntegerArray poss_y, boolean keep_order, int window_size) {
		List<IntegerArray> ret = Generics.newLinkedList();
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
		return new IntegerArrayMatrix(ret);
	}

	public static PostingList findCollocations(PostingList pl_x, PostingList pl_y, boolean keep_order, int window_size) throws Exception {
		IntegerArray dseqs_x = pl_x.getDocSeqs();
		IntegerArray dseqs_y = pl_y.getDocSeqs();
		IntegerArrayMatrix posData_x = pl_x.getPosData();
		IntegerArrayMatrix posData_y = pl_y.getPosData();
		IntegerArrayMatrix posData_x_min = pl_x.getEndPosData();

		List<Integer> dseqs_xy = Generics.newLinkedList();
		List<IntegerArray> posData_xy = Generics.newLinkedList();
		List<IntegerArray> posData_xy_min = Generics.newLinkedList();

		int i = 0;
		int j = 0;

		while (i < dseqs_x.size() && j < dseqs_y.size()) {
			int dseq_x = dseqs_x.get(i);
			int dseq_y = dseqs_y.get(j);

			if (dseq_x == dseq_y) {
				IntegerArray poss_x = posData_x.get(i);
				IntegerArray poss_y = posData_y.get(j);
				IntegerArray poss_x_min = posData_x_min == null ? new IntegerArray() : posData_x_min.get(i);

				IntegerArrayMatrix ps = findCollocations(poss_x, poss_y, keep_order, window_size);

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

		PostingList ret = new PostingList(pl_y.getWord(), new IntegerArray(dseqs_xy), new IntegerArrayMatrix(posData_xy));

		IntegerArrayMatrix m = new IntegerArrayMatrix(posData_xy_min);
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

	// public static void main(String[] args) throws Exception {
	// System.out.println("process begins.");
	//
	// String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR,
	// MIRPath.TREC_CDS_2016_COL_DC_DIR };
	//
	// for (int i = 0; i < dirs.length; i++) {
	// System.out.printf("read [%s]\n", dirs[i]);
	// InvertedIndex ii = new InvertedIndex(dirs[i]);
	// DocumentCollection dc = new DocumentCollection(dirs[i]);
	// Vocab vocab = dc.getVocab();
	//
	// Counter<Integer> c = Generics.newCounter(vocab.size());
	//
	// int s = 0;
	//
	// while (s < ii.size()) {
	// int e = Math.min(ii.size(), s + 100);
	// List<SPostingList> pls = ii.getPostingLists(s, e);
	//
	// for (SPostingList pl : pls) {
	// int w = pl.getWord();
	// String word = vocab.getObject(w);
	// c.setCount(w, pl.size());
	// }
	// s = e;
	// }
	//
	// // for (int w = 0; w < ii.size(); w++) {
	// // SPostingList pl = ii.getPostingList(w);
	// // String word = vocab.getObject(w);
	// //
	// // c.setCount(w, pl.size());
	// //
	// // // System.out.printf("word=[%s], %s\n", word, pl);
	// // }
	//
	// IntegerArray ws = new IntegerArray(c.getSortedKeys());
	//
	// for (int j = 0; j < ws.size() && j < 50; j++) {
	// int w = ws.get(j);
	// String word = vocab.getObject(w);
	// SPostingList pl = ii.getPostingList(w);
	// System.out.printf("word=[%s], %s\n", word, pl);
	// }
	//
	// }
	//
	// System.out.println("process ends.");
	// }

	private Map<Integer, PostingList> cache = Generics.newWeakHashMap();

	private File dataDir;

	private FileChannel fc;

	private LongArray starts;

	private IntegerArray lens;

	private int doc_cnt = 0;

	private boolean encode = false;

	public InvertedIndex(FileChannel fc, LongArray starts, IntegerArray lens, int doc_cnt) {
		this(fc, starts, lens, doc_cnt, Generics.newWeakHashMap());
	}

	public InvertedIndex(FileChannel fc, LongArray starts, IntegerArray lens, int doc_cnt, Map<Integer, PostingList> cache) {
		this.fc = fc;
		this.starts = starts;
		this.lens = lens;
		this.doc_cnt = doc_cnt;
		this.cache = cache;
	}

	public InvertedIndex(String dataDir) throws Exception {
		this.dataDir = new File(dataDir);

		{
			FileChannel fc = FileUtils.openFileChannel(new File(dataDir, META_NAME), "r");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			doc_cnt = ByteArrayUtils.toInteger(data.get(i++));
			starts = DataCompression.decodeToLongArray(data.get(i++));
			lens = DataCompression.decodeToIntegerArray(data.get(i++));
			DataCompression.decodeGaps(starts);
		}

		fc = FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r");
	}

	public void close() throws Exception {
		fc.close();
	}

	public InvertedIndex copyShallow() throws Exception {
		return new InvertedIndex(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), starts, lens, doc_cnt, cache);
	}

	public int getDocCnt() {
		return doc_cnt;
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public PostingList getPostingList(int w) throws Exception {
		if (w < 0 || w >= starts.size()) {
			return null;
		}

		PostingList ret = null;

		synchronized (cache) {
			ret = cache.get(w);
		}

		if (ret == null) {
			synchronized (fc) {
				long start = starts.get(w);
				if (start < 0) {
					return null;
				}
				fc.position(start);
				ret = PostingList.readPostingList(fc, encode);
			}

			synchronized (cache) {
				cache.put(w, ret);
			}
		}
		return ret;
	}

	/**
	 * @param Q
	 * @param keep_order
	 * @param window_size
	 * @return
	 * @throws Exception
	 */
	public PostingList getPostingList(IntegerArray Q, boolean keep_order, int window_size) throws Exception {
		PostingList pl_x = getPostingList(Q.get(0));
		PostingList ret = pl_x;

		if (Q.size() > 1) {
			PostingList tmp = pl_x;

			int i = 0;
			while (++i < Q.size()) {
				int y = Q.get(i);
				PostingList pl_y = getPostingList(y);
				PostingList pl_xy = findCollocations(pl_x, pl_y, keep_order, window_size);

				if (pl_xy.size() == 0) {
					break;
				}

				pl_x = pl_xy;
			}

			if (i == Q.size()) {
				ret = pl_x;
				ret.setWord(-1);

				IntegerArrayMatrix posData_min = ret.getEndPosData();
				IntegerArrayMatrix posData = ret.getPosData();

				ret.setPosData(posData_min);
				ret.setEndPosData(posData);
			}
		}

		return ret;
	}

	public List<PostingList> getPostingLists(Collection<Integer> Q) throws Exception {
		return getPostingLists(new IntegerArray(Q));
	}

	public List<PostingList> getPostingLists(int i, int j) throws Exception {
		int size = j - i;

		Map<Integer, PostingList> found = Generics.newHashMap(size);
		Set<Integer> notFound = Generics.newHashSet(size);

		for (int k = i; k < j; k++) {
			PostingList p = null;

			synchronized (cache) {
				p = cache.get(k);
			}

			if (p == null) {
				notFound.add(k);
			} else {
				found.put(k, p);
			}
		}

		List<PostingList> ret = Generics.newArrayList(size);

		if (found.size() == 0) {
			fc.position(starts.get(i));
			ByteArray data = FileUtils.readByteArray(fc, ArrayMath.sum(lens.values(), i, j));
			ByteBufferWrapper buf = new ByteBufferWrapper(data);

			for (int k = i; k < j; k++) {
				ByteArrayMatrix sub = buf.readByteArrayMatrix();
				PostingList pl = PostingList.toPostingList(sub, encode);
				ret.add(pl);

				synchronized (cache) {
					cache.put(k, pl);
				}
			}
		} else {
			if (found.size() != size) {
				for (int k : notFound) {
					found.put(k, getPostingList(k));
				}
			}
			for (int k = i; k < j; k++) {
				ret.add(found.get(k));
			}
		}

		return ret;
	}

	public List<PostingList> getPostingLists(IntegerArray Q) throws Exception {
		List<PostingList> ret = Generics.newArrayList(Q.size());
		for (int w : Q) {
			ret.add(getPostingList(w));
		}
		return ret;
	}

	public int size() {
		return starts.size();
	}

}
