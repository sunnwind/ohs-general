package ohs.ml.glove;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.ByteSize;
import ohs.utils.ByteSize.Type;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class CooccurrenceCounter {

	public class CountWorker implements Callable<Integer> {

		private DocumentCollection dc;

		private List<FileChannel> fcs;

		private int[][] ranges;

		private AtomicInteger range_cnt;

		private Timer timer;

		private ByteBufferWrapper buf = new ByteBufferWrapper(max_buf_size);

		public CountWorker(DocumentCollection lsc, List<FileChannel> fcs, int[][] ranges, AtomicInteger range_cnt, Timer timer) {
			super();
			this.dc = lsc;
			this.fcs = fcs;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.timer = timer;
		}

		private boolean accept(int w) {
			boolean ret = true;
			if (toPrune.size() > 0 && toPrune.contains(w)) {
				ret = false;
			}
			return ret;
		}

		@Override
		public Integer call() throws Exception {
			int range_loc = 0;

			CounterMap<Integer, Integer> cm1 = Generics.newCounterMap();
			int buf_size = 0;

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] range = ranges[range_loc];
				List<Pair<String, IntegerArray>> ps = dc.getRange(range);

				for (int i = 0; i < ps.size(); i++) {
					IntegerArrayMatrix doc = DocumentCollection.toMultiSentences(ps.get(i).getSecond());
					CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();

					for (IntegerArray sent : doc) {
						for (int j = 1; j < sent.size(); j++) {
							int start = Math.max(j - window_size, 0);
							int w_center = sent.get(j);
							String word1 = dc.getVocab().getObject(w_center);
							
							if (!accept(w_center)) {
								continue;
							}

							for (int k = start; k < j; k++) {
								int w_left = sent.get(k);
								String word2 = dc.getVocab().getObject(w_left);
								
								if (!accept(w_left)) {
									continue;
								}
								double dist = (j - k);
								double cocnt = 1f / dist;
								cm2.incrementCount(w_center, w_left, cocnt);
							}
						}
					}

					int size = cm2.size();
					for (Entry<Integer, Counter<Integer>> e : cm2.getEntrySet()) {
						size += Integer.BYTES;
						size += e.getValue().size() * (Integer.BYTES + Double.BYTES);
					}
					buf_size += size;

					cm1.incrementAll(cm2);

					if (buf_size >= max_buf_size) {
						write(cm1);
						if (symmetric) {
							write(cm1.invert());
						}
						cm1.clear();
						buf_size = 0;
					}
				}

				int prog = BatchUtils.progress(range_loc + 1, ranges.length);

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, range_loc + 1, ranges.length, timer.stop());
				}
			}

			if (cm1.size() > 0) {
				write(cm1);
				if (symmetric) {
					write(cm1.invert());
				}
				cm1.clear();
			}
			return 0;
		}

		@Override
		protected void finalize() throws Throwable {
			dc.close();
		}

		private void write(CounterMap<Integer, Integer> cm) throws Exception {
			for (Entry<Integer, Counter<Integer>> e : cm.getEntrySet()) {
				int w1 = e.getKey();
				Counter<Integer> c = e.getValue();

				IntegerArray idxs = new IntegerArray(c.size());
				DoubleArray vals = new DoubleArray(c.size());

				for (Entry<Integer, Double> e2 : c.entrySet()) {
					idxs.add(e2.getKey());
					vals.add(e2.getValue());
				}

				int buf_size = ByteArrayUtils.sizeOfByteBuffer(idxs) + ByteArrayUtils.sizeOfByteBuffer(vals) + Integer.BYTES * 2;
				buf.ensureCapacity(buf_size);

				buf.clear();
				buf.write(buf_size - Integer.BYTES);
				buf.write(w1);
				buf.write(idxs);
				buf.write(vals);

				int loc = w1 % output_file_size;
				FileChannel fc = fcs.get(loc);

				synchronized (fc) {
					FileUtils.write(buf.getByteBuffer(), fc);
				}
			}
		}
	}

	public class MergeWorker implements Callable<Integer> {
		private AtomicInteger file_cnt;

		private List<File> files;

		public MergeWorker(List<File> files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				CounterMap<Integer, Integer> cm = Generics.newCounterMap(dc.getVocab().size());

				File file = files.get(file_loc);
				FileChannel fc = FileUtils.openFileChannel(file.getPath(), "rw");

				while (fc.position() < fc.size()) {
					ByteArray data = FileUtils.readByteArray(fc);
					ByteBufferWrapper buf = new ByteBufferWrapper(data);
					long pos = fc.position();

					int w1 = buf.readInteger();
					IntegerArray idxs = buf.readIntegerArray();
					DoubleArray vals = buf.readDoubleArray();

					Counter<Integer> c = cm.getCounter(w1);

					for (int i = 0; i < idxs.size(); i++) {
						c.incrementCount(idxs.get(i), vals.get(i));
					}
				}
				fc.close();
				file.delete();
				new SparseMatrix(cm).writeObject(file.getPath());

				// ByteArrayMatrix data = new ByteArrayMatrix(cm.size());
				// IntegerArray ws1 = new IntegerArray(cm.keySet());
				// ws1.sort(false);
				//
				// data.add(ByteArrayUtils.toByteArray(ws1));
				//
				// for (int w1 : ws1) {
				// Counter<Integer> c = cm.getCounter(w1);
				//
				// IntegerArray ws2 = new IntegerArray(c.keySet());
				// ws2.sort(false);
				//
				// DoubleArray vals = new DoubleArray(c.size());
				//
				// for (int w2 : ws2) {
				// vals.add(c.getCount(w2));
				// }
				//
				// data.add(ByteArrayUtils.toByteArray(ws2));
				// data.add(ByteArrayUtils.toByteArray(vals));
				// }

				// fc = FileUtils.openFileChannel(file.getPath(), "rw");
				// FileUtils.write(data, fc);
				// fc.close();
			}
			return 0;
		}
	}

	public static final String NAME = "cocnt";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// String dirPath = MIRPath.OHSUMED_DIR;

		// String[] inDirs = { MIRPath.OHSUMED_COL_DC_DIR,
		// MIRPath.CLEF_EH_2014_COL_DC_DIR, MIRPath.TREC_GENO_2007_COL_DC_DIR,
		// MIRPath.TREC_CDS_2014_COL_DC_DIR, MIRPath.TREC_CDS_2016_COL_DC_DIR,
		// MIRPath.WIKI_COL_DC_DIR };
		String[] dataDirs = { MIRPath.OHSUMED_COL_DC_DIR };

		for (int i = 0; i < dataDirs.length; i++) {
			String dataDir = dataDirs[i];
			CooccurrenceCounter cc = new CooccurrenceCounter(dataDir, null);
			cc.setWindowSize(10);
			cc.setCountThreadSize(1);
			cc.setMergeThreadSize(1);
			cc.setSymmetric(true);
			cc.setOutputFileSize(100);
			cc.count();
		}

		System.out.println("process ends.");
	}

	private int max_buf_size = (int) new ByteSize(32, Type.MEGA).getBytes();

	private int window_size = 10;

	private int count_thread_size = 10;

	private int merge_thread_size = 1;

	private int output_file_size = 100;

	private int batch_size = 200;

	private boolean symmetric = false;

	private Set<Integer> toPrune;

	private DocumentCollection dc;

	private File dataDir;

	private File outDir;

	public CooccurrenceCounter(String dataDir, Set<String> stopwords) throws Exception {
		this.dataDir = new File(dataDir);

		outDir = new File(dataDir, NAME);

		dc = new DocumentCollection(dataDir);

		toPrune = Generics.newHashSet();

		Vocab vocab = dc.getVocab();

		if (stopwords != null) {
			for (String word : stopwords) {
				int w = vocab.indexOf(word.toLowerCase());
				if (w > -1) {
					toPrune.add(w);
				}
			}
		}

		int unk = vocab.indexOf(Vocab.SYM.UNK.getText());

		if (unk != -1) {
			toPrune.add(unk);
		}

		for (String word : vocab) {
			if (word.matches("^\\p{Punct}+$")) {
				toPrune.add(vocab.indexOf(word));
			}
		}
	}

	public void count() throws Exception {
		createFiles();
		mergeFiles();
	}

	private void createFiles() throws Exception {
		System.out.println("create files.");
		Timer timer = Timer.newTimer();

		FileUtils.deleteFilesUnder(outDir);

		List<FileChannel> fcs = Generics.newArrayList(output_file_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		int[][] ranges = BatchUtils.getBatchRanges(dc.size(), batch_size);

		AtomicInteger range_cnt = new AtomicInteger();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(count_thread_size);

		try {
			for (int i = 0; i < output_file_size; i++) {
				DecimalFormat df = new DecimalFormat("00000");
				fcs.add(FileUtils.openFileChannel(new File(outDir, String.format("%s.ser", df.format(i))), "rw"));
			}

			for (int i = 0; i < count_thread_size; i++) {
				fs.add(tpe.submit(new CountWorker(dc.copyShallow(), fcs, ranges, range_cnt, timer)));
			}

			for (int j = 0; j < fs.size(); j++) {
				fs.get(j).get();
			}

		} finally {
			tpe.shutdown();

			for (FileChannel fc : fcs) {
				fc.close();
			}
		}
	}

	public DocumentCollection getDocumentCollection() {
		return dc;
	}

	private void mergeFiles() throws Exception {
		System.out.println("merge files.");

		List<File> files = FileUtils.getFilesUnder(outDir);

		AtomicInteger file_cnt = new AtomicInteger();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(merge_thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		for (int i = 0; i < merge_thread_size; i++) {
			fs.add(tpe.submit(new MergeWorker(files, file_cnt)));
		}

		for (int j = 0; j < fs.size(); j++) {
			fs.get(j).get();
		}
		fs.clear();
		tpe.shutdown();
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setCountThreadSize(int thread_size) {
		this.count_thread_size = thread_size;
	}

	public void setMergeThreadSize(int thread_size) {
		this.merge_thread_size = thread_size;
	}

	public void setOutputFileSize(int output_file_size) {
		this.output_file_size = output_file_size;
	}

	public void setSymmetric(boolean symmetric) {
		this.symmetric = symmetric;
	}

	public void setWindowSize(int window_size) {
		this.window_size = window_size;
	}

}
