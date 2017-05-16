package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.xalan.xsltc.compiler.sym;

import java.util.Set;

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.ByteSize;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class DocumentCollection {

	public static final String DATA_NAME = "int_docs.ser";

	public static final String META_NAME = "int_docs_meta.ser";

	public static final String VOCAB_NAME = "vocab.ser";

	public static final String VECTOR_NAME = "doc_vecs.ser";

	public static final String VECTOR_META_NAME = "doc_vecs_meta.ser";

	public static final int SENT_END = -1;

	public static String getText(Vocab vocab, IntegerArrayMatrix doc) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < doc.size(); i++) {
			IntegerArray sent = doc.get(i);
			sb.append(StrUtils.join(" ", vocab.getObjects(sent.values())));
			if (i != doc.size() - 1) {
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR, MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR,
					MIRPath.WIKI_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR };

			for (int j = 1; j < dirs.length; j++) {
				String dir = dirs[j];

				System.out.println(dir);

				DocumentCollection dc = new DocumentCollection(dir);
				int[][] ranges = BatchUtils.getBatchRanges(dc.size(), 500);

				Timer timer = Timer.newTimer();

				for (int i = 0; i < ranges.length; i++) {
					Timer timer2 = Timer.newTimer();
					dc.getRange(ranges[i][0], ranges[i][1], true);
					// System.out.println(timer2.stop());
				}

				System.out.println(timer.stop());

				dc = new DocumentCollection(dir);
				timer = Timer.newTimer();

				Timer timer2 = Timer.newTimer();

				for (int i = 0; i < dc.size(); i++) {
					if ((i + 1) % 10000 == 0 || i == dc.size() - 1) {
						// System.out.println(timer2.stop());
						timer2 = Timer.newTimer();
					}
					dc.get(i);
				}

				System.out.println(timer.stop());
				System.out.println();
			}
		}

		// {
		// String[] dirs = { MIRPath.OHSUMED_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.TREC_CDS_2014_COL_DC_DIR };
		//
		// for (int j = 0; j < dirs.length; j++) {
		// String dir = dirs[j];
		// if (j != 2) {
		// continue;
		// }
		//
		// DocumentCollection ldc = new DocumentCollection(dir);
		// System.out.printf("[%s]\n", dir);
		// System.out.println(ldc.size());
		//
		// List<Pair<String, String>> ps = ldc.getText(0, 5);
		//
		// for (int i = 0; i < ps.size(); i++) {
		// Pair<String, String> p = ps.get(i);
		// String text = p.getSecond();
		// String[] sents = text.split("\n");
		// System.out.println(p.getFirst());
		// System.out.println(StrUtils.join("\n", sents, 0, 10));
		// System.out.println("----------------------------------");
		// }
		// }
		// }

		// {
		// DocumentCollection ldc = new
		// DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		// System.out.println(DocumentCollection.getText(ldc.vocab,
		// ldc.getSents(0).getSecond()));
		// System.out.println(ldc.size());
		// }

		// {
		// DocumentCollection ldc = new
		// DocumentCollection(MIRPath.BIOASQ_COL_DC_DIR);
		// // System.out.println(DocumentCollection.getText(ldc.vocab,
		// ldc.getSents(0).getSecond()));
		// // System.out.println(ldc.size());
		//
		// System.out.println(1f * ldc.getLength() / Integer.MAX_VALUE * 100);
		// }
		//
		// {
		// DocumentCollection ldc = new
		// DocumentCollection(MIRPath.CLUEWEB_COL_DC_DIR);
		// // System.out.println(DocumentCollection.getText(ldc.vocab,
		// ldc.getSents(0).getSecond()));
		// // System.out.println(ldc.size());
		//
		// System.out.println(1f * ldc.getLength() / Integer.MAX_VALUE * 100);
		// }

		System.out.println("process ends.");
	}

	public static Vocab readVocab(String inFile) throws Exception {
		FileChannel fc = FileUtils.openFileChannel(inFile, "r");
		ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
		fc.close();

		int i = 0;
		IntegerArray wordCnts = DataCompression.decodeToIntegerArray(data.get(i++));
		IntegerArray docFreqs = DataCompression.decodeToIntegerArray(data.get(i++));
		int vocab_size = ByteArrayUtils.toInteger(data.get(i++).values());
		int doc_cnt = ByteArrayUtils.toInteger(data.get(i++).values());
		List<String> words = Generics.newArrayList(vocab_size);

		while (i < data.size()) {
			String s = DataCompression.decodeToString(data.get(i++));
			for (String word : s.split("\t")) {
				words.add(word);
			}
		}

		DataCompression.decodeGaps(wordCnts.values());
		DataCompression.decodeGaps(docFreqs.values());

		Vocab vocab = new Vocab(words);
		vocab.setWordCnts(wordCnts);
		vocab.setDocFreqs(docFreqs);
		vocab.setDocCnt(doc_cnt);
		return vocab;
	}

	public static IntegerArrayMatrix toMultiSentences(IntegerArray d) {
		List<Integer> cnts = Generics.newLinkedList();
		int w_cnt = 0;
		for (int w : d) {
			if (w == SENT_END) {
				cnts.add(w_cnt);
				w_cnt = 0;
			} else {
				w_cnt++;
			}
		}
		cnts.add(w_cnt);

		IntegerArrayMatrix ret = new IntegerArrayMatrix(cnts.size());

		for (int cnt : cnts) {
			ret.add(new IntegerArray(cnt));
		}

		int sent_cnt = 0;
		for (int w : d) {
			if (w == SENT_END) {
				sent_cnt++;
			} else {
				ret.get(sent_cnt).add(w);
			}
		}
		ret.trimToSize();
		return ret;
	}

	public static IntegerArray toSingleSentence(IntegerArrayMatrix doc) {
		IntegerArray ret = new IntegerArray(doc.sizeOfEntries() + doc.size());
		for (int i = 0; i < doc.size(); i++) {
			IntegerArray sent = doc.get(i);
			for (int w : sent) {
				ret.add(w);
			}
			if (i != doc.size() - 1) {
				ret.add(SENT_END);
			}
		}
		ret.trimToSize();
		return ret;
	}

	private Map<Integer, Pair<String, IntegerArray>> cache = Generics.newWeakHashMap();;

	private Vocab vocab = new Vocab();

	private long len_d_min;

	private long len_d_max;

	private double len_d_avg;

	private long len_c;

	private FileChannel fc;

	private File dataDir;

	private LongArray starts;

	private IntegerArray lens;

	private boolean encode = false;

	private boolean use_cache = true;

	public DocumentCollection() {

	}

	public void setUseCache(boolean use_cache) {
		this.use_cache = use_cache;
	}

	public DocumentCollection(FileChannel fc, LongArray starts, IntegerArray lens, Vocab vocab,
			Map<Integer, Pair<String, IntegerArray>> cache) throws Exception {
		this.fc = fc;
		this.starts = starts;
		this.lens = lens;
		this.vocab = vocab;
		this.cache = cache;
	}

	public DocumentCollection(String dataDir) throws Exception {
		this.dataDir = new File(dataDir);

		{
			FileChannel fc = FileUtils.openFileChannel(new File(dataDir, META_NAME), "r");
			ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);
			fc.close();

			int i = 0;
			starts = DataCompression.decodeToLongArray(data.get(i++));
			lens = DataCompression.decodeToIntegerArray(data.get(i++));
			len_d_min = ByteArrayUtils.toInteger(data.get(i++));
			len_d_max = ByteArrayUtils.toInteger(data.get(i++));
			len_c = ByteArrayUtils.toLong(data.get(i++));
			len_d_avg = ByteArrayUtils.toDouble(data.get(i++));
			encode = data.get(i++).get(0) == 1 ? true : false;

			DataCompression.decodeGaps(starts.values());
		}

		vocab = readVocab(new File(dataDir, VOCAB_NAME).getPath());

		fc = FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r");
	}

	public void close() throws Exception {
		fc.close();
	}

	public DocumentCollection copyShallow() throws Exception {
		return new DocumentCollection(FileUtils.openFileChannel(new File(dataDir, DATA_NAME), "r"), starts, lens, vocab, cache);
	}

	public Pair<String, IntegerArray> get(int i) throws Exception {
		Pair<String, IntegerArray> ret = null;

		if (use_cache) {
			synchronized (cache) {
				ret = cache.get(i);
			}
		}

		if (ret == null) {
			ByteArrayMatrix data = null;

			synchronized (fc) {
				long start = starts.get(i);
				fc.position(start);
				int len = lens.get(i);
				data = new ByteBufferWrapper(FileUtils.readByteArray(fc, len)).readByteArrayMatrix();
			}

			String docid = null;
			IntegerArray d = null;

			if (encode) {
				docid = DataCompression.decodeToString(data.get(0));
				d = DataCompression.decodeToIntegerArray(data.get(1));
			} else {
				docid = new String(data.get(0).values());
				d = ByteArrayUtils.toIntegerArray(data.get(1));
			}
			ret = Generics.newPair(docid, d);

			if (use_cache) {
				synchronized (cache) {
					cache.put(i, ret);
				}
			}
		}

		return ret;
	}

	public List<Pair<String, IntegerArray>> get(int[] is) throws Exception {
		List<Pair<String, IntegerArray>> ret = Generics.newArrayList(is.length);
		for (int i : is) {
			ret.add(get(i));
		}
		return ret;
	}

	public double getAvgDocLength() {
		return len_d_avg;
	}

	public ByteSize getByteSize() {
		long bytes = vocab.byteSize().getBytes();
		bytes += Integer.BYTES * len_c;
		return new ByteSize(bytes);
	}

	public File getDataDir() {
		return dataDir;
	}

	public String getDocId(int dseq) throws Exception {
		return get(dseq).getFirst();
	}

	public List<String> getDocIds() throws Exception {
		List<String> ret = Generics.newArrayList(size());
		int[][] ranges = BatchUtils.getBatchRanges(size(), 1000);
		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];
			List<Pair<String, IntegerArray>> ls = getRange(range);
			for (int j = 0; j < ls.size(); j++) {
				int dseq = range[0] + j;
				ret.add(ls.get(j).getFirst());
			}
		}
		return ret;
	}

	public int getDocLength(int i) {
		return lens.get(i);
	}

	public SparseVector getDocVector(int i) throws Exception {
		IntegerArray d = get(i).getSecond();
		Counter<Integer> c = Generics.newCounter(d.size());
		for (int w : d) {
			if (w == SENT_END) {
				continue;
			}
			c.incrementCount(w, 1);
		}
		return new SparseVector(c);
	}

	public SparseMatrix getDocVectors(int[] is) throws Exception {
		Map<Integer, SparseVector> m = Generics.newHashMap(is.length);
		for (int i : is) {
			m.put(i, getDocVector(i));
		}
		return new SparseMatrix(m);
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public long getLength() {
		return len_c;
	}

	public long getMaxDocLength() {
		return len_d_max;
	}

	public long getMinDocLength() {
		return len_d_min;
	}

	public List<Pair<String, IntegerArray>> getRange(int i, int j, boolean use_cache) throws Exception {
		int size = j - i;
		List<Pair<String, IntegerArray>> ret = Generics.newArrayList(size);

		if (use_cache) {
			Map<Integer, Pair<String, IntegerArray>> found = Generics.newHashMap(size);
			Set<Integer> notFound = Generics.newHashSet(size);

			for (int k = i; k < j; k++) {
				Pair<String, IntegerArray> p = null;

				synchronized (cache) {
					p = cache.get(k);
				}

				if (p == null) {
					notFound.add(k);
				} else {
					found.put(k, p);
				}
			}

			if (found.size() == 0) {
				ByteArray data = null;

				synchronized (fc) {
					fc.position(starts.get(i));
					data = FileUtils.readByteArray(fc, ArrayMath.sum(lens.values(), i, j));
				}

				ByteBufferWrapper buf = new ByteBufferWrapper(data);

				for (int k = i; k < j; k++) {
					ByteArrayMatrix sub = buf.readByteArrayMatrix();

					String docid = null;
					IntegerArray d = null;
					int u = 0;
					if (encode) {
						docid = DataCompression.decodeToString(sub.get(u++));
						d = DataCompression.decodeToIntegerArray(sub.get(u++));
					} else {
						docid = new String(sub.get(u++).values());
						d = ByteArrayUtils.toIntegerArray(sub.get(u++));
					}
					Pair<String, IntegerArray> p = Generics.newPair(docid, d);
					ret.add(p);

					synchronized (cache) {
						cache.put(k, p);
					}
				}
			} else {
				if (found.size() != size) {
					for (int k : notFound) {
						found.put(k, get(k));
					}
				}
				for (int k = i; k < j; k++) {
					ret.add(found.get(k));
				}
			}
		} else {
			ByteArray data = null;

			synchronized (fc) {
				fc.position(starts.get(i));
				data = FileUtils.readByteArray(fc, ArrayMath.sum(lens.values(), i, j));
			}

			ByteBufferWrapper buf = new ByteBufferWrapper(data);

			for (int k = i; k < j; k++) {
				ByteArrayMatrix sub = buf.readByteArrayMatrix();

				String docid = null;
				IntegerArray d = null;
				int u = 0;
				if (encode) {
					docid = DataCompression.decodeToString(sub.get(u++));
					d = DataCompression.decodeToIntegerArray(sub.get(u++));
				} else {
					docid = new String(sub.get(u++).values());
					d = ByteArrayUtils.toIntegerArray(sub.get(u++));
				}
				Pair<String, IntegerArray> p = Generics.newPair(docid, d);
				ret.add(p);
			}
		}
		return ret;
	}

	public List<Pair<String, IntegerArray>> getRange(int[] range) throws Exception {
		return getRange(range[0], range[1], true);
	}

	public SparseMatrix getRangeDocVectors(int i, int j) throws Exception {
		List<Pair<String, IntegerArray>> ps = getRange(i, j, false);
		List<Integer> idxs = Generics.newArrayList(ps.size());
		List<SparseVector> dvs = Generics.newArrayList(ps.size());

		int k = i;
		for (Pair<String, IntegerArray> p : ps) {
			IntegerArray d = p.getSecond();
			Counter<Integer> c = Generics.newCounter(d.size());
			for (int w : d) {
				if (w == SENT_END) {
					continue;
				}
				c.incrementCount(w, 1);
			}

			idxs.add(k++);
			dvs.add(new SparseVector(c));
		}
		return new SparseMatrix(idxs, dvs);
	}

	public Pair<String, IntegerArrayMatrix> getSents(int i) throws Exception {
		Pair<String, IntegerArray> p = get(i);
		return Generics.newPair(p.getFirst(), toMultiSentences(p.getSecond()));
	}

	public LongArray getStarts() {
		return starts;
	}

	// public LongArray getPositions(int i) throws Exception {
	// long start = starts.get(i);
	// fc.position(start);
	//
	// ByteArrayMatrix data = ByteArrayUtils.readByteArrayMatrix(fc);
	// String docid = null;
	// IntegerArray d = null;
	//
	// int j = 0;
	// if (encode) {
	// docid = DataCompression.decodeToString(data.get(j++));
	// d = DataCompression.decodeToIntegerArray(data.get(j++));
	// } else {
	// docid = new String(data.get(j++).values());
	// d = ByteArrayUtils.toIntegerArray(data.get(j++));
	// }
	//
	// long tmp_start = start + 3 * Integer.BYTES + data.get(0).size();
	// LongArray ret = new LongArray(d.size());
	//
	// for (int k = 0; k < d.size(); k++) {
	// ret.add(tmp_start + k * Integer.BYTES);
	// }
	//
	// return ret;
	// }

	public Pair<String, String> getText(int i) throws Exception {
		Pair<String, IntegerArray> p = get(i);
		IntegerArrayMatrix doc = toMultiSentences(get(i).getSecond());

		StringBuffer sb = new StringBuffer();
		for (int j = 0; j < doc.size(); j++) {
			IntegerArray sent = doc.get(j);
			sb.append(StrUtils.join(" ", vocab.getObjects(sent.values())));
			if (j != doc.size() - 1) {
				sb.append("\n");
			}
		}
		return Generics.newPair(p.getFirst(), sb.toString());
	}

	public List<Pair<String, String>> getText(int i, int j) throws Exception {
		List<Pair<String, String>> ret = Generics.newArrayList(j - i);
		for (Pair<String, IntegerArray> p : getRange(i, j, true)) {
			IntegerArrayMatrix doc = toMultiSentences(p.getSecond());
			StringBuffer sb = new StringBuffer();
			for (int k = 0; k < doc.size(); k++) {
				IntegerArray sent = doc.get(k);
				sb.append(StrUtils.join(" ", vocab.getObjects(sent.values())));
				if (k != doc.size() - 1) {
					sb.append("\n");
				}
			}
			ret.add(Generics.newPair(p.getFirst(), sb.toString()));
		}
		return ret;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public List<String> getWords(int i) throws Exception {
		IntegerArray d = get(i).getSecond();
		List<String> ret = Generics.newArrayList(d.size());
		for (int j = 0; j < d.size(); j++) {
			int w = d.get(j);

			if (w == DocumentCollection.SENT_END) {
				ret.add("\n");
			} else {
				ret.add(vocab.getObject(w));
			}
		}
		return ret;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Collection Info ]\n");
		sb.append(String.format("docs:\t[%d]\n", size()));
		sb.append(String.format("col len:\t[%d]\n", len_c));
		sb.append(String.format("min doc len:\t[%d]\n", len_d_min));
		sb.append(String.format("max doc len:\t[%d]\n", len_d_max));
		sb.append(String.format("avg doc len:\t[%f]\n", len_d_avg));
		sb.append(String.format("mem:\t%s", getByteSize().toString()));
		return sb.toString();
	}

	public int size() {
		return starts.size();
	}

	public String toString() {
		return info();
	}

}
