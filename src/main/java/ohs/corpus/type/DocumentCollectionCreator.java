package ohs.corpus.type;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.ListList;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.ByteSize;
import ohs.utils.ByteSize.Type;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class DocumentCollectionCreator {

	class CountWorker implements Callable<Pair<Counter<String>, Counter<String>>> {

		private AtomicInteger doc_cnt;

		private AtomicInteger range_cnt;

		private IntegerArrayMatrix ranges;

		private RawDocumentCollection rdc;

		private Counter<String> docFreqs;

		private Counter<String> wordCnts;

		private Timer timer;

		public CountWorker(RawDocumentCollection rdc, IntegerArrayMatrix ranges, AtomicInteger range_cnt,
				AtomicInteger file_cnt, Counter<String> wordCnts, Counter<String> docFreqs, Timer timer) {
			this.rdc = rdc;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.doc_cnt = file_cnt;
			this.wordCnts = wordCnts;
			this.docFreqs = docFreqs;
			this.timer = timer;
		}

		@Override
		public Pair<Counter<String>, Counter<String>> call() throws Exception {
			int range_loc = 0;

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.size()) {
				IntegerArray range = ranges.get(range_loc);

				ListList<String> data = rdc.getValues(range.values());
				Counter<String> cnts = Generics.newCounter();
				Counter<String> freqs = Generics.newCounter();

				for (int i = 0; i < data.size(); i++) {
					int dseq = range.get(0) + i;
					int type = rdc.getColSeq(dseq);
					IntegerArray target_locs = target_loc_data.get(type);

					List<String> vals = data.get(i, false);
					Counter<String> c = Generics.newCounter();

					for (int loc : target_locs) {
						String p = vals.get(loc);

						for (String sent : p.split("[\\n]+")) {
							if (sent.length() == 0) {
								continue;
							}

							for (String word : st.tokenize(sent)) {
								if (word.length() == 0) {
									continue;
								}
								if (word.length() == 0) {
									continue;
								}
								c.incrementCount(word, 1);
							}
						}
					}

					for (Entry<String, Double> e : c.entrySet()) {
						cnts.incrementCount(e.getKey(), e.getValue());
						freqs.incrementCount(e.getKey(), 1);
					}

					int j = doc_cnt.incrementAndGet();
					int prog = BatchUtils.progress(j, rdc.size());
					if (prog > 0) {
						System.out.printf("[%s percent, %d/%d, %s]\n", prog, j, rdc.size(), timer.stop());
					}
				}

				synchronized (wordCnts) {
					wordCnts.incrementAll(cnts);
				}

				synchronized (docFreqs) {
					docFreqs.incrementAll(freqs);
				}
			}
			return Generics.newPair(docFreqs, wordCnts);
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			rdc.close();
		}
	}

	public class IndexWorker implements Callable<Integer> {

		private Vocab vocab;

		private IntegerArrayMatrix ranges;

		private AtomicInteger range_cnt;

		private RawDocumentCollection rdc;

		private AtomicInteger doc_cnt;

		private AtomicInteger file_cnt;

		private Timer timer;

		private ByteBufferWrapper buf = new ByteBufferWrapper(FileUtils.DEFAULT_BUF_SIZE);

		public IndexWorker(RawDocumentCollection rdc, Vocab vocab, IntegerArrayMatrix ranges, AtomicInteger range_cnt,
				AtomicInteger doc_cnt, AtomicInteger file_cnt, Timer timer) {
			this.rdc = rdc;
			this.vocab = vocab;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.doc_cnt = doc_cnt;
			this.file_cnt = file_cnt;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int range_loc = 0;

			List<ByteArrayMatrix> docs = Generics.newArrayList(batch_size);

			int unk = vocab.indexOf(Vocab.SYM.UNK.getText());

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.size()) {
				IntegerArray range = ranges.get(range_loc);
				IntegerArrayMatrix subranges = new IntegerArrayMatrix(
						BatchUtils.getBatchRanges(range.get(1) - range.get(0), 100));

				File outFile = new File(tmpDir, new DecimalFormat("00000000").format(range_loc) + ".ser");
				FileChannel out = FileUtils.openFileChannel(outFile, "rw");

				for (int i = 0; i < subranges.size(); i++) {
					IntegerArray subrange = subranges.get(i);
					subrange.set(0, range.get(0) + subrange.get(0));
					subrange.set(1, range.get(0) + subrange.get(1));

					ListList<String> ps = rdc.getValues(subrange.values());

					for (int j = 0; j < ps.size(); j++) {
						int dseq = subrange.get(0) + j;
						int type = rdc.getColSeq(dseq);
						IntegerArray target_locs = target_loc_data.get(type);
						List<String> vals = ps.get(j);

						List<IntegerArray> sents = Generics.newArrayList(100);

						for (int loc : target_locs) {
							String p = vals.get(loc);

							for (String s : p.split("[\\n]+")) {
								if (s.length() == 0) {
									continue;
								}
								List<String> words = st.tokenize(s);
								IntegerArray sent = new IntegerArray(vocab.indexesOf(words, unk));
								sents.add(sent);
							}
						}
						IntegerArrayMatrix doc = new IntegerArrayMatrix(sents);
						IntegerArray d = DocumentCollection.toSingleSentence(doc);

						int len_d = doc.sizeOfEntries();

						lens_d.set(dseq, len_d);

						String docid = "";

						if (docid_locs.get(type) != -1) {
							docid = vals.get(docid_locs.get(type));
						}

						if (docs.size() >= batch_size) {
							write(docs, out);
						}

						ByteArrayMatrix data = new ByteArrayMatrix(2);
						data.add(encode ? DataCompression.encode(docid) : new ByteArray(docid.getBytes()));
						data.add(encode ? DataCompression.encode(d) : ByteArrayUtils.toByteArray(d));
						docs.add(data);

						int k = doc_cnt.incrementAndGet();
						int prog = BatchUtils.progress(k, rdc.size());
						if (prog > 0) {
							System.out.printf("[%d percent, %d/%d, %s]\n", prog, k, rdc.size(), timer.stop());
						}
					}

					if (docs.size() > 0) {
						write(docs, out);
					}
				}

				if (docs.size() > 0) {
					write(docs, out);
				}

				out.close();
			}

			return null;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			rdc.close();
		}

		private void write(List<ByteArrayMatrix> docs, FileChannel fc) throws Exception {
			for (ByteArrayMatrix doc : docs) {
				FileUtils.write(doc, fc);
			}
			docs.clear();
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentCollectionCreator dcc = new DocumentCollectionCreator();
		dcc.setBatchSize(300);
		dcc.setCountingThreadSize(10);
		dcc.setIndexingThreadSize(5);
		// dcc.setReuseVocab(true);
		dcc.setEncode(false);

		// dcc.create(MIRPath.OHSUMED_COL_DC_DIR, 1, new int[] { 3, 5 });
		// dcc.create(MIRPath.CLEF_EH_2014_COL_DC_DIR, 0, new int[] { 3 });
		// dcc.create(MIRPath.TREC_GENO_2007_COL_DC_DIR, 0, new int[] { 1 });
		// dcc.create(MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR, 0, new int[] { 2, 3, 4
		// });
		// dcc.create(MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR, 0, new int[] { 1, 2, 3,
		// 4, 5 });
		// dcc.create(MIRPath.TREC_CDS_2014_COL_DC_DIR, 0, new int[] { 1, 2, 3 });
		// dcc.create(MIRPath.TREC_CDS_2016_COL_DC_DIR, 0, new int[] { 1, 2, 3 });
		// dcc.create(MIRPath.BIOASQ_COL_DC_DIR, 0, new int[] { 4, 5 });
		// dcc.create(MIRPath.WIKI_COL_DC_DIR, 0, new int[] { 3 });
		// dcc.create("../../data/medical_ir/scopus/col/dc/", 0, new int[] { 1, 2 });
		// dcc.create(MIRPath.CLUEWEB_COL_DC_DIR, 0, new int[] { 1 });

		// scc.setMinDocFreq(0);
		// scc.create(MIRPath.MESH_COL_LINE_DIR, 0, new int[] { 1
		// },MIRPath.MESH_COL_DC_DIR);

		// {
		//
		// // inDirNames.add(MIRPath.OHSUMED_COL_DC_DIR);
		// // inDirNames.add(MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR);
		// // inDirNames.add(MIRPath.TREC_PM_2017_COL_CLINICAL_DC_DIR);
		// // inDirNames.add(MIRPath.TREC_GENO_2007_COL_DC_DIR);
		// // inDirNames.add(MIRPath.BIOASQ_COL_DC_DIR);
		// // inDirNames.add(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		// // inDirNames.add(MIRPath.WIKI_COL_DC_DIR);
		//
		// int[] docid_locs = { 1, 0, 0, 0, 0, 0, -1 };
		// int[][] target_loc_data = new int[docid_locs.length][];
		// target_loc_data[0] = new int[] { 3, 5 };
		// target_loc_data[1] = new int[] { 2, 3, 4 };
		// target_loc_data[2] = new int[] { 1, 2, 3, 4, 5 };
		// target_loc_data[3] = new int[] { 1 };
		// target_loc_data[4] = new int[] { 4, 5 };
		// target_loc_data[5] = new int[] { 1, 2, 3 };
		// target_loc_data[6] = new int[] { 2, 3 };
		//
		// // int[] docid_locs = { 1, 0 };
		// // int[][] target_loc_data = new int[docid_locs.length][];
		// // target_loc_data[0] = new int[] { 3, 5 };
		// // target_loc_data[1] = new int[] { 1 };
		//
		// // dcc.create(MIRPath.DATA_DIR + "merged/col/dc/", docid_locs,
		// target_loc_data);
		// }

		{

			String[] attrs = { "type", "cn", "kor_kwds", "eng_kwds", "kor_title", "eng_title", "kor_abs", "eng_abs",
					"kor_pos_kwds", "kor_pos_title", "kor_pos_abs" };
			dcc.setStringTokenizer(new KoreanPosTokenizer());
			dcc.create(KPPath.COL_DC_DIR, 0, new int[] { 9, 10 });
		}

		System.out.println("process ends.");
	}

	public static void writeVocab(Vocab vocab, String outFile) throws Exception {
		DataCompression.encodeGaps(vocab.getCounts().values());
		DataCompression.encodeGaps(vocab.getDocFreqs().values());

		ByteArrayMatrix data = new ByteArrayMatrix(4);
		data.add(DataCompression.encode(vocab.getCounts()));
		data.add(DataCompression.encode(vocab.getDocFreqs()));
		data.add(ByteArrayUtils.toByteArray(vocab.size()));
		data.add(ByteArrayUtils.toByteArray(vocab.getDocCnt()));

		int word_cnt_per_block = 200000;
		int[][] ranges = BatchUtils.getBatchRanges(vocab.size(), word_cnt_per_block);

		for (int k = 0; k < ranges.length; k++) {
			int[] range = ranges[k];
			StringBuffer sb = new StringBuffer();
			int start = range[0];
			int end = range[1];
			for (int w = start; w < end; w++) {
				sb.append(vocab.getObject(w));
				if (w != end - 1) {
					sb.append("\t");
				}
			}
			data.add(DataCompression.encode(sb.toString()));
		}

		FileUtils.delete(outFile);

		FileChannel fc = FileUtils.openFileChannel(outFile, "rw");
		FileUtils.write(data, fc);
		fc.close();
	}

	private StringTokenizer st = new EnglishTokenizer(new EnglishNormalizer(), false);

	private boolean add_unknown_tag = true;

	private boolean add_boundary_tags = false;

	private boolean reuse_vocab = false;

	private boolean encode = false;

	private IntegerArrayMatrix target_loc_data;

	private int counting_thread_size = 5;

	private int indexing_thread_size = 1;

	private int max_word_len = 100;

	private IntegerArray docid_locs;

	private int min_doc_freq = 5;

	private int min_word_cnt = 5;

	private int batch_size = 500;

	private int max_buf_size = (int) new ByteSize(64, Type.MEGA).getBytes();

	private File dataDir;

	private File tmpDir;

	private RawDocumentCollection rdc;

	private IntegerArray lens_d;

	public DocumentCollectionCreator() {

	}

	private String[] addBoundaryTags(String[] words) {
		String[] ret = new String[words.length + 2];
		ret[0] = Vocab.SYM.START.getText();

		for (int i = 0; i < words.length; i++) {
			ret[i + 1] = words[i];
		}
		ret[ret.length - 1] = Vocab.SYM.END.getText();
		return ret;
	}

	public void create(String dataDir, int docid_loc, int[] target_locs) throws Exception {
		System.out.printf("create document collection at [%s]\n", dataDir);

		this.dataDir = new File(dataDir);
		this.docid_locs = new IntegerArray(new int[] { docid_loc });
		this.target_loc_data = new IntegerArrayMatrix();
		this.target_loc_data.add(new IntegerArray(target_locs));

		this.tmpDir = new File(dataDir, "tmp/");

		rdc = new RawDocumentCollection(dataDir);

		lens_d = new IntegerArray(new int[rdc.size()]);

		File vocabFile = new File(dataDir, DocumentCollection.VOCAB_NAME);

		Vocab vocab = null;

		if (reuse_vocab && vocabFile.exists()) {
			vocab = DocumentCollection.readVocab(vocabFile.getPath());
		} else {
			vocab = createVocab();
			writeVocab(vocab, vocabFile.getPath());
		}

		indexDocs(vocab);
		mergeFiles(vocab);
	}

	public void create(String dataDir, int[] docid_locs, int[][] target_locs) throws Exception {
		System.out.printf("create document collection at [%s]\n", dataDir);

		this.dataDir = new File(dataDir);
		this.docid_locs = new IntegerArray(docid_locs);
		this.target_loc_data = new IntegerArrayMatrix(target_locs);
		this.tmpDir = new File(dataDir, "tmp/");

		rdc = new RawDocumentCollection(dataDir);

		lens_d = new IntegerArray(new int[rdc.size()]);

		File vocabFile = new File(dataDir, DocumentCollection.VOCAB_NAME);

		Vocab vocab = null;

		if (reuse_vocab && vocabFile.exists()) {
			vocab = DocumentCollection.readVocab(vocabFile.getPath());
		} else {
			vocab = createVocab();
			writeVocab(vocab, vocabFile.getPath());
		}

		indexDocs(vocab);
		mergeFiles(vocab);
	}

	public Vocab createVocab() throws Exception {
		System.out.println("create vocab.");

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(counting_thread_size);

		List<Future<Pair<Counter<String>, Counter<String>>>> fs = Generics.newArrayList(counting_thread_size);

		AtomicInteger doc_cnt = new AtomicInteger(0);
		Timer timer = Timer.newTimer();

		IntegerArrayMatrix ranges = getRanges(batch_size);
		AtomicInteger range_cnt = new AtomicInteger(0);

		int init_size = 1000000;

		Counter<String> docFreqs = Generics.newCounter(init_size);
		Counter<String> wordCnts = Generics.newCounter(init_size);

		for (int i = 0; i < counting_thread_size; i++) {
			fs.add(tpe
					.submit(new CountWorker(rdc.copyShallow(), ranges, range_cnt, doc_cnt, wordCnts, docFreqs, timer)));
		}

		for (int i = 0; i < counting_thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		tpe.shutdown();

		return createVocab(docFreqs, wordCnts);
	}

	private Vocab createVocab(Counter<String> docFreqs, Counter<String> wordCnts) {
		int vocab_size_old = wordCnts.size();

		Counter<String> c1 = Generics.newCounter();
		Counter<String> c2 = Generics.newCounter();

		double unk_word_cnt = 0;
		double unk_doc_freq = 0;

		for (String word : docFreqs.keySet()) {
			double word_cnt = wordCnts.getCount(word);
			double doc_freq = docFreqs.getCount(word);

			if (word.length() > max_word_len || word_cnt < min_word_cnt) {
				unk_word_cnt += word_cnt;
				unk_doc_freq += doc_freq;
			} else {
				c1.setCount(word, doc_freq);
				c2.setCount(word, word_cnt);
			}
		}

		docFreqs = c1;
		wordCnts = c2;

		docFreqs.setCount(Vocab.SYM.UNK.getText(), unk_doc_freq);
		wordCnts.setCount(Vocab.SYM.UNK.getText(), unk_word_cnt);

		Vocab vocab = new Vocab();

		for (String word : docFreqs.getSortedKeys(true)) {
			vocab.add(word);
		}

		IntegerArray doc_freqs = new IntegerArray(new int[vocab.size()]);
		IntegerArray word_cnts = new IntegerArray(new int[vocab.size()]);

		for (String word : vocab) {
			int w = vocab.indexOf(word);
			doc_freqs.set(w, (int) docFreqs.getCount(word));
			word_cnts.set(w, (int) wordCnts.getCount(word));
		}

		vocab.setWordCnts(word_cnts);
		vocab.setDocFreqs(doc_freqs);
		vocab.setDocCnt(rdc.size());

		System.out.printf("vocab size: %d -> %d\n", vocab_size_old, vocab.size());

		return vocab;
	}

	private IntegerArrayMatrix getRanges(int batch_size) {
		IntegerArrayMatrix trs = rdc.getTypeRanges();
		List<IntPair> tmp = Generics.newArrayList(rdc.size() / batch_size);
		int dseq = 0;

		for (IntegerArray tr : trs) {
			int last = tr.get(1);

			while (dseq < last) {
				int s = dseq;
				int e = Math.min(dseq + batch_size, last);
				tmp.add(new IntPair(s, e));
				dseq = e;
			}
		}

		int[][] ret = new int[tmp.size()][2];

		for (int i = 0; i < tmp.size(); i++) {
			IntPair p = tmp.get(i);
			ret[i] = new int[] { p.getFirst(), p.getSecond() };
		}
		return new IntegerArrayMatrix(ret);
	}

	private int index_batch_size = 10000;

	private void indexDocs(Vocab vocab) throws Exception {
		System.out.println("index docs.");

		FileUtils.deleteFilesUnder(tmpDir);
		tmpDir.mkdirs();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(indexing_thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(indexing_thread_size);

		IntegerArrayMatrix ranges = getRanges(100000);

		AtomicInteger range_cnt = new AtomicInteger(0);

		AtomicInteger doc_cnt = new AtomicInteger(0);

		AtomicInteger file_cnt = new AtomicInteger(0);

		Timer timer = Timer.newTimer();

		for (int i = 0; i < indexing_thread_size; i++) {
			fs.add(tpe.submit(new IndexWorker(rdc.copyShallow(), vocab, ranges, range_cnt, doc_cnt, file_cnt, timer)));
		}

		for (int i = 0; i < indexing_thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		tpe.shutdown();
	}

	private IntegerArray indexWords(Vocab vocab, String[] words) {
		if (add_boundary_tags) {
			words = addBoundaryTags(words);
		}

		IntegerArray sent = null;

		if (add_unknown_tag) {
			sent = vocab.indexesOf(words, Vocab.SYM.UNK.ordinal());
		} else {
			sent = vocab.indexesOfKnown(words);
		}
		sent.trimToSize();
		return sent;
	}

	private void mergeFiles(Vocab vocab) throws Exception {
		System.out.println("merge files.");
		Timer timer = Timer.newTimer();

		long len_c = 0;
		int len_d_min = Integer.MAX_VALUE;
		int len_d_max = 0;
		double len_d_avg = 0;
		int doc_cnt = 0;

		for (int len_d : lens_d) {
			len_c += len_d;
			len_d_min = Math.min(len_d_min, len_d);
			len_d_max = Math.max(len_d_max, len_d);
		}
		len_d_avg = 1d * len_c / rdc.size();

		File outFile1 = new File(dataDir, DocumentCollection.META_NAME);
		File outFile2 = new File(dataDir, DocumentCollection.DATA_NAME);

		FileUtils.delete(outFile1.getPath());
		FileUtils.delete(outFile2.getPath());

		FileChannel out1 = FileUtils.openFileChannel(outFile1, "rw");
		FileChannel out2 = FileUtils.openFileChannel(outFile2, "rw");

		LongArray starts = new LongArray(rdc.size());
		IntegerArray lens = new IntegerArray(rdc.size());

		for (File file : FileUtils.getFilesUnder(tmpDir)) {
			List<ByteArrayMatrix> docs = Generics.newArrayList(batch_size);

			FileChannel in = FileUtils.openFileChannel(file.getPath(), "r");
			while (in.position() < in.size()) {
				if (docs.size() == 1000) {
					for (ByteArrayMatrix doc : docs) {
						long[] info = FileUtils.write(doc, out2);

						starts.add(info[0]);
						lens.add((int) info[1]);

						int prog = BatchUtils.progress(++doc_cnt, rdc.size());

						if (prog > 0) {
							System.out.printf("[%d percent, %d/%d, %s]\n", prog, doc_cnt, rdc.size(), timer.stop());
						}
					}
					docs.clear();
				} else {
					ByteArrayMatrix doc = FileUtils.readByteArrayMatrix(in);
					docs.add(doc);
				}
			}
			in.close();
			file.delete();

			for (ByteArrayMatrix doc : docs) {
				long[] info = FileUtils.write(doc, out2);

				starts.add(info[0]);
				lens.add((int) info[1]);

				int prog = BatchUtils.progress(++doc_cnt, rdc.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, doc_cnt, rdc.size(), timer.stop());
				}
			}
		}

		len_d_avg = 1d * len_c / starts.size();

		DataCompression.encodeGaps(starts);

		ByteArrayMatrix data = new ByteArrayMatrix();
		data.add(DataCompression.encode(starts));
		data.add(DataCompression.encode(lens));
		data.add(ByteArrayUtils.toByteArray(len_d_min));
		data.add(ByteArrayUtils.toByteArray(len_d_max));
		data.add(ByteArrayUtils.toByteArray(len_c));
		data.add(ByteArrayUtils.toByteArray(len_d_avg));
		data.add(new ByteArray(new byte[] { (byte) (encode ? 1 : 0) }));

		FileUtils.write(data, out1);

		out1.close();
		out2.close();

		FileUtils.deleteFilesUnder(tmpDir);
	}

	public void setAddBoundaryTags(boolean add_boundary_tags) {
		this.add_boundary_tags = add_boundary_tags;
	}

	public void setAddUnkwonTag(boolean add_unkwon_tag) {
		this.add_unknown_tag = add_unkwon_tag;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setCountingThreadSize(int thread_size) {
		this.counting_thread_size = thread_size;
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setIndexingThreadSize(int thread_size) {
		this.indexing_thread_size = thread_size;
	}

	public void SetMaxBufferSize(int max_buf_size) {
		this.max_buf_size = max_buf_size;
	}

	public void setMaxWordLen(int max_word_len) {
		this.max_word_len = max_word_len;
	}

	public void setMinDocFreq(int min_doc_freq) {
		this.min_doc_freq = min_doc_freq;
	}

	public void setMinWordCnt(int min_word_cnt) {
		this.min_word_cnt = min_word_cnt;
	}

	public void setReuseVocab(boolean reuse_vocab) {
		this.reuse_vocab = reuse_vocab;
	}

	public void setStringTokenizer(StringTokenizer et) {
		this.st = et;
	}

}
