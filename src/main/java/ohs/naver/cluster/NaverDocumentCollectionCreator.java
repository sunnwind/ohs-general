package ohs.naver.cluster;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentCollectionCreator;
import ohs.corpus.type.RawDocumentCollection;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ml.neuralnet.com.BatchUtils;
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

public class NaverDocumentCollectionCreator {

	class CountWorker implements Callable<Pair<Counter<String>, Counter<String>>> {

		private AtomicInteger doc_cnt;

		private AtomicInteger range_cnt;

		private int[][] ranges;

		private RawDocumentCollection rdc;

		private Counter<String> docFreqs;

		private Counter<String> wordCnts;

		private Timer timer;

		public CountWorker(RawDocumentCollection rdc, int[][] ranges, AtomicInteger range_cnt, AtomicInteger file_cnt,
				Counter<String> wordCnts, Counter<String> docFreqs, Timer timer) {
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

			Counter<String> cnts = Generics.newCounter();
			Counter<String> freqs = Generics.newCounter();
			Counter<String> c = Generics.newCounter();

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] range = ranges[range_loc];

				ListList<String> data = rdc.getValues(range);

				cnts.clear();
				freqs.clear();

				for (int i = 0; i < data.size(); i++) {
					List<String> vals = data.get(i, false);
					c.clear();

					for (int loc : target_locs) {
						String p = vals.get(loc);

						for (String s : p.split("[\n]+")) {
							if (s.length() == 0) {
								continue;
							}

							s = sn.normalize(s);

							if (s.length() == 0) {
								continue;
							}

							for (String word : s.split(" ")) {
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

		private int[][] ranges;

		private AtomicInteger range_cnt;

		private RawDocumentCollection rdc;

		private AtomicInteger doc_cnt;

		private Timer timer;

		private ByteBufferWrapper buf = new ByteBufferWrapper(FileUtils.DEFAULT_BUF_SIZE);

		public IndexWorker(RawDocumentCollection rdc, Vocab vocab, int[][] ranges, AtomicInteger range_cnt, AtomicInteger doc_cnt,
				Timer timer) {
			this.rdc = rdc;
			this.vocab = vocab;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.doc_cnt = doc_cnt;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int range_loc = 0;

			List<ByteArrayMatrix> docs = Generics.newLinkedList();

			int unk = vocab.indexOf(Vocab.SYM.UNK.getText());

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] range = ranges[range_loc];
				int data_size = range[1] - range[0];

				File outFile = new File(tmpDir, new DecimalFormat("00000000").format(range_loc) + ".ser");
				FileChannel fc = FileUtils.openFileChannel(outFile, "rw");

				int[][] subranges = BatchUtils.getBatchRanges(data_size, batch_size);

				for (int i = 0; i < subranges.length; i++) {
					int[] subrange = subranges[i];
					for (int j = 0; j < subrange.length; j++) {
						subrange[j] += range[0];
					}

					ListList<String> ps = rdc.getValues(subrange);

					for (int j = 0, dseq = subrange[0]; j < ps.size(); j++, dseq++) {
						List<String> vals = ps.get(j);

						List<IntegerArray> sents = Generics.newLinkedList();

						for (int loc : target_locs) {
							String p = vals.get(loc);

							for (String s : p.split("[\n]+")) {
								if (s.length() == 0) {
									continue;
								}
								s = sn.normalize(s);
								if (s.length() == 0) {
									continue;
								}

								IntegerArray sent = vocab.indexesOf(s.split(" "), unk);

								sents.add(sent);
							}
						}
						IntegerArrayMatrix doc = new IntegerArrayMatrix(sents);
						IntegerArray d = DocumentCollection.toSingleSentence(doc);

						int len_d = doc.sizeOfEntries();

						lens_d.set(dseq, len_d);

						String docid = "";

						if (docid_loc != -1) {
							docid = vals.get(docid_loc);
						}

						if (docs.size() >= batch_size) {
							write(docs, fc);
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

					for (int j = subrange[0]; j < subrange[1]; j++) {

					}
				}

				if (docs.size() > 0) {
					write(docs, fc);
				}

				fc.close();
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
				FileUtils.write(doc, buf, fc);
			}
			docs.clear();
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentCollectionCreator dcc = new DocumentCollectionCreator();
		dcc.setCreateVocabBatchSize(200);
		dcc.setCountingThreadSize(10);
		dcc.setIndexingThreadSize(5);

		dcc.setMinWordCnt(0);
		dcc.create("../../data/naver_news/col/dc/", -1, new int[] { 0 });

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

	private int batch_size = 200;

	private StringNormalizer sn = new SimpleStringNormalizer(false);

	private boolean add_unknown_tag = true;

	private boolean add_boundary_tags = false;

	private boolean reuse_vocab = false;

	private boolean encode = false;

	private int[] target_locs;

	private int counting_thread_size = 5;

	private int indexing_thread_size = 1;

	private int max_word_len = 100;

	private int docid_loc = -1;

	private int min_doc_freq = 5;

	private int min_word_cnt = 5;

	private int create_vocab_batch_size = 500;

	private int max_buf_size = (int) new ByteSize(64, Type.MEGA).getBytes();

	private File dataDir;

	private File tmpDir;

	private RawDocumentCollection rdc;

	private IntegerArray lens_d;

	public NaverDocumentCollectionCreator() {

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
		this.docid_loc = docid_loc;
		this.target_locs = target_locs;
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

	private Vocab createVocab() throws Exception {
		System.out.println("create vocab.");

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(counting_thread_size);

		List<Future<Pair<Counter<String>, Counter<String>>>> fs = Generics.newArrayList(counting_thread_size);

		AtomicInteger doc_cnt = new AtomicInteger(0);
		Timer timer = Timer.newTimer();

		int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), create_vocab_batch_size);
		AtomicInteger range_cnt = new AtomicInteger(0);

		// int init_size = 150000000;

		int init_size = 1000000;

		Counter<String> docFreqs = Generics.newCounter(init_size);
		Counter<String> wordCnts = Generics.newCounter(init_size);

		for (int i = 0; i < counting_thread_size; i++) {
			fs.add(tpe.submit(new CountWorker(rdc.copyShallow(), ranges, range_cnt, doc_cnt, wordCnts, docFreqs, timer)));
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

		int[] doc_freqs = new int[vocab.size()];
		int[] word_cnts = new int[vocab.size()];

		for (String word : vocab) {
			int w = vocab.indexOf(word);
			doc_freqs[w] = (int) docFreqs.getCount(word);
			word_cnts[w] = (int) wordCnts.getCount(word);
		}

		vocab.setWordCnts(new IntegerArray(word_cnts));
		vocab.setDocFreqs(new IntegerArray(doc_freqs));
		vocab.setDocCnt(rdc.size());

		System.out.printf("vocab size: %d -> %d\n", vocab_size_old, vocab.size());

		return vocab;
	}

	private void indexDocs(Vocab vocab) throws Exception {
		System.out.println("index docs.");

		FileUtils.deleteFilesUnder(tmpDir);
		tmpDir.mkdirs();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(indexing_thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(indexing_thread_size);

		int batch_size = 100;

		int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), rdc.size() / batch_size);

		AtomicInteger range_cnt = new AtomicInteger(0);

		AtomicInteger doc_cnt = new AtomicInteger(0);

		Timer timer = Timer.newTimer();

		for (int i = 0; i < indexing_thread_size; i++) {
			fs.add(tpe.submit(new IndexWorker(rdc.copyShallow(), vocab, ranges, range_cnt, doc_cnt, timer)));
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
		len_d_avg = 1f * len_c / rdc.size();

		File outFile1 = new File(dataDir, DocumentCollection.META_NAME);
		File outFile2 = new File(dataDir, DocumentCollection.DATA_NAME);

		FileUtils.delete(outFile1.getPath());
		FileUtils.delete(outFile2.getPath());

		FileChannel out1 = FileUtils.openFileChannel(outFile1, "rw");
		FileChannel out2 = FileUtils.openFileChannel(outFile2, "rw");

		LongArray starts = new LongArray(rdc.size());
		IntegerArray lens = new IntegerArray(rdc.size());

		List<ByteArrayMatrix> docs = Generics.newLinkedList();

		ByteBufferWrapper buf = new ByteBufferWrapper(max_buf_size);

		for (File file : FileUtils.getFilesUnder(tmpDir)) {
			FileChannel in = FileUtils.openFileChannel(file.getPath(), "r");

			while (in.position() < in.size()) {
				if (docs.size() == batch_size) {
					for (ByteArrayMatrix doc : docs) {
						long[] info = FileUtils.write(doc, buf, out2);

						starts.add(info[0]);
						lens.add((int) info[1]);

						int prog = BatchUtils.progress(++doc_cnt, rdc.size());
						if (prog > 0) {
							System.out.printf("[%d percent, %s]\n", prog, timer.stop());
						}
					}
					docs.clear();
				}

				ByteArrayMatrix doc = FileUtils.readByteArrayMatrix(in);
				docs.add(doc);
			}

			in.close();
			file.delete();
		}

		if (docs.size() > 0) {
			for (ByteArrayMatrix doc : docs) {
				long[] info = FileUtils.write(doc, buf, out2);
				starts.add(info[0]);
				lens.add((int) info[1]);

				int prog = BatchUtils.progress(++doc_cnt, rdc.size());
				if (prog > 0) {
					System.out.printf("[%d percent, %s]\n", prog, timer.stop());
				}
			}
			docs.clear();
		}

		len_d_avg = 1f * len_c / starts.size();

		DataCompression.encodeGaps(starts);

		ByteArrayMatrix data = new ByteArrayMatrix();
		data.add(DataCompression.encode(starts));
		data.add(DataCompression.encode(lens));
		data.add(ByteArrayUtils.toByteArray(len_d_min));
		data.add(ByteArrayUtils.toByteArray(len_d_max));
		data.add(ByteArrayUtils.toByteArray(len_c));
		data.add(ByteArrayUtils.toByteArray(len_d_avg));
		data.add(new ByteArray(new byte[] { (byte) (encode ? 1 : 0) }));

		FileUtils.write(data, buf, out1);

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

	public void setCountingThreadSize(int thread_size) {
		this.counting_thread_size = thread_size;
	}

	public void setCreateVocabBatchSize(int batch_size) {
		this.create_vocab_batch_size = batch_size;
	}

	public void setDocIdLoc(int docid_loc) {
		this.docid_loc = docid_loc;
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setIndexingThreadSize(int thread_size) {
		this.indexing_thread_size = thread_size;
	}

	public void SetMaxBufferSize(int max_buf_size) {
		this.batch_size = max_buf_size;
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

	public void setStringNormalizer(StringNormalizer sn) {
		this.sn = sn;
	}

	private void writeDocs(List<ByteArrayMatrix> docs, ByteBuffer buf, FileChannel fc) {

	}
}
