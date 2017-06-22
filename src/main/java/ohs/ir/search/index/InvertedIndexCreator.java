package ohs.ir.search.index;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * 
 * http://a07274.tistory.com/281
 * 
 * http://eincs.com/2009/08/java-nio-bytebuffer-channel/
 * 
 * @author ohs
 */
public class InvertedIndexCreator {

	public class MergeWorker implements Callable<Integer> {

		private List<File> files;

		private AtomicInteger file_cnt;

		private Timer timer;

		public MergeWorker(List<File> files, AtomicInteger file_cnt, Timer timer) {
			this.files = files;
			this.file_cnt = file_cnt;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;

			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File inFile = files.get(file_loc);

				FileChannel fc = FileUtils.openFileChannel(inFile, "rw");
				int cnt = 0;

				ListMapMap<Integer, Integer, Integer> lmm = Generics.newListMapMap();

				while (fc.position() < fc.size()) {
					ByteArrayMatrix data = FileUtils.readByteArrayMatrix(fc);

					ListMapMap<Integer, Integer, Integer> lmm2 = Generics.newListMapMap(200);

					for (ByteArray sub : data) {
						ByteBufferWrapper buf = new ByteBufferWrapper(sub);

						int w = buf.readInteger();
						IntegerArray dseqs = buf.readIntegerArray();
						IntegerArrayMatrix posData = buf.readIntegerArrayMatrix();

						ListMap<Integer, Integer> lm = lmm2.get(w, true);

						for (int k = 0; k < dseqs.size(); k++) {
							int dseq = dseqs.get(k);
							IntegerArray poss = posData.get(k);
							List<Integer> l = lm.get(dseq, true);
							for (int pos : poss) {
								l.add(pos);
							}
						}
					}

					for (int w : lmm2.keySet()) {
						ListMap<Integer, Integer> lm = lmm2.get(w);
						for (int dseq : lm.keySet()) {
							List<Integer> poss = lm.get(dseq);
							lmm.get(w, dseq, true).addAll(poss);
						}
					}
					lmm2 = null;

					cnt++;
				}
				fc.close();
				// inFile.delete();

				// System.out.printf("[read, %s, %d, %s]\n", inFile.getName(), cnt, timer.stop());

				File outFile = new File(inFile.getParentFile(), "m_" + inFile.getName());

				fc = FileUtils.openFileChannel(outFile, "rw");

				IntegerArray ws = new IntegerArray(lmm.keySet());
				ws.sort(false);

				cnt = 0;

				List<PostingList> pls = Generics.newArrayList(ws.size());
				int buf_size = 0;
				int max_buf_size = FileUtils.DEFAULT_BUF_SIZE;

				for (int w : ws) {
					ListMap<Integer, Integer> lm = lmm.get(w);
					IntegerArray dseqs = new IntegerArray(lm.keySet());
					dseqs.sort(false);

					IntegerArrayMatrix posData = new IntegerArrayMatrix(dseqs.size());

					for (int dseq : dseqs) {
						IntegerArray poss = new IntegerArray(lm.get(dseq));
						poss.sort(false);
						posData.add(poss);
					}

					if (buf_size > max_buf_size) {
						write(pls, fc);
						buf_size = 0;
					}

					PostingList pl = new PostingList(w, dseqs, posData);
					pls.add(pl);

					buf_size += PostingList.sizeOfByteBuffer(pl);
					cnt++;
				}
				lmm = null;

				if (pls.size() > 0) {
					write(pls, fc);
					buf_size = 0;
				}

				pls = null;

				// System.out.printf("[write, %s, %d, %s]\n", inFile.getName(), cnt, timer.stop());

				fc.close();

				int prog = BatchUtils.progress(file_loc + 1, files.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, file_loc + 1, files.size(), timer.stop());
				}
			}

			return null;
		}

		private void write(List<PostingList> pls, FileChannel fc) throws Exception {
			for (PostingList pl : pls) {
				PostingList.writePostingList(pl, fc);
			}
			pls.clear();
		}

	}

	public class PostingWorker implements Callable<Integer> {

		private AtomicInteger range_cnt;

		private AtomicInteger doc_cnt;

		private IntegerArrayMatrix ranges;

		private Timer timer;

		private DocumentCollection dc;

		public PostingWorker(DocumentCollection dc, AtomicInteger range_cnt, AtomicInteger doc_cnt, IntegerArrayMatrix ranges,
				Timer timer) {
			super();
			this.dc = dc;
			this.range_cnt = range_cnt;
			this.doc_cnt = doc_cnt;
			this.ranges = ranges;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int rloc = 0;

			while ((rloc = range_cnt.getAndIncrement()) < ranges.size()) {
				IntegerArray range = ranges.get(rloc);
				int batch_size = range.get(1) - range.get(0);
				IntegerArrayMatrix subranges = new IntegerArrayMatrix(BatchUtils.getBatchRanges(batch_size, 500));
				String fileName = tmpDir + new DecimalFormat("00000").format(rloc) + ".ser";
				FileChannel fc = FileUtils.openFileChannel(fileName, "rw");

				for (int m = 0; m < subranges.size(); m++) {
					IntegerArray subrange = subranges.get(m);
					List<Pair<String, IntegerArray>> ps = dc.getRange(subrange.get(0), subrange.get(1), false);
					ListMapMap<Integer, Integer, Integer> lmm = Generics.newListMapMap(200);

					for (int i = 0; i < ps.size(); i++) {
						int dseq = subrange.get(0) + i;
						Pair<String, IntegerArray> p = ps.get(i);
						IntegerArray d = p.getSecond();
						for (int pos = 0; pos < d.size(); pos++) {
							int w = d.get(pos);
							if (w == DocumentCollection.SENT_END) {
								continue;
							}
							lmm.put(w, dseq, pos);
						}
						int dloc = doc_cnt.incrementAndGet();
						int prog = BatchUtils.progress(dloc, dc.size());

						if (prog > 0) {
							System.out.printf("[%d percent, %d/%d, %s]\n", prog, dloc, dc.size(), timer.stop());
						}
					}

					if (lmm.size() > 0) {
						write(fc, lmm);
					}
				}
			}

			return null;
		}

		private void write(FileChannel fc, ListMapMap<Integer, Integer, Integer> lmm) throws Exception {
			IntegerArray ws = new IntegerArray(lmm.keySet());
			ws.sort(false);

			for (int w : ws) {
				ListMap<Integer, Integer> lm = lmm.get(w);
				IntegerArray dseqs = new IntegerArray(lm.keySet());
				dseqs.sort(false);

				IntegerArrayMatrix posData = new IntegerArrayMatrix(dseqs.size());
				int max_poss_len = 0;

				for (int dseq : dseqs) {
					IntegerArray poss = new IntegerArray(lm.get(dseq));
					poss.sort(false);
					posData.add(poss);
					max_poss_len = Math.max(poss.size(), max_poss_len);
				}

				PostingList pl = new PostingList(w, dseqs, posData);
				PostingList.writePostingList(pl, fc);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		InvertedIndexCreator iic = new InvertedIndexCreator();
		iic.setBatchSize(200);
		iic.setPostingThreadSize(10);
		iic.setMergingThreadSize(2);
		iic.setEncode(true);
		iic.create(MIRPath.OHSUMED_COL_DC_DIR);
		// iic.create(MIRPath.TREC_CDS_2014_COL_DC_DIR);
		// iic.create(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		// iic.create(MIRPath.BIOASQ_COL_DC_DIR);
		// iic.create(MIRPath.WIKI_COL_DC_DIR);
		// iic.create(MIRPath.CLUEWEB_COL_DC_DIR);

		// iic.create(MIRPath.DATA_DIR + "merged/col/dc/");

		// FrequentPhraseDetector.main(args);

		// iic.create(MIRPath.TREC_CDS_2014_COL_DC_DIR);
		// iic.create(MIRPath.CLEF_EH_2014_COL_DC_DIR);
		// iic.create(MIRPath.TREC_GENO_2007_COL_DC_DIR);
		// iic.create(MIRPath.MESH_COL_DC_DIR);
		//

		// InvertedIndex ii = new InvertedIndex(MIRPath.OHSUMED_COL_DC_DIR);
		//
		// FileChannel fc = ii.getFileChannel();
		//
		// int size = 0;
		//
		// while (fc.position() < fc.size()) {
		// SPostingList pl = SPostingList.readPostingList(fc, true);
		// size++;
		// }

		// System.out.println(size);

		System.out.println("process ends.");
	}

	private int posting_thread_size = 1;

	private int batch_size = 500;

	private DocumentCollection dc;

	private File dataDir;

	private File tmpDir;

	private int merge_thread_size = 1;

	private boolean encode = false;

	public InvertedIndexCreator() {

	}

	public void create(String dataDir) throws Exception {
		System.out.printf("create inverted index at [%s]\n", dataDir);

		this.dataDir = new File(dataDir);
		this.tmpDir = new File(dataDir, "tmp");
		dc = new DocumentCollection(dataDir);

		createPostingListFiles();
		mergePostingListFiles();
		mergePostingLists();
	}

	private void createPostingListFiles() throws Exception {
		System.out.println("create posting list files.");

		Timer timer = Timer.newTimer();

		FileUtils.deleteFilesUnder(tmpDir);

		tmpDir.mkdirs();

		IntegerArrayMatrix ranges = new IntegerArrayMatrix(BatchUtils.getBatchRanges(dc.size(), 100000));

		AtomicInteger range_cnt = new AtomicInteger(0);

		AtomicInteger doc_cnt = new AtomicInteger(0);

		posting_thread_size = Math.min(ranges.size(), posting_thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(posting_thread_size);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(posting_thread_size);

		try {

			for (int i = 0; i < posting_thread_size; i++) {
				fs.add(tpe.submit(new PostingWorker(dc.copyShallow(), range_cnt, doc_cnt, ranges, timer)));
			}
			for (int i = 0; i < posting_thread_size; i++) {
				fs.get(i).get();
			}
		} finally {
			tpe.shutdown();
		}
	}

	private void mergePostingListFiles() throws Exception {
		System.out.println("merge posting list files.");
		Timer timer = Timer.newTimer();

		Counter<File> c = Generics.newCounter();

		for (File file : FileUtils.getFilesUnder(tmpDir)) {
			if (file.getName().startsWith("m_")) {
				file.delete();
				continue;
			}
			c.setCount(file, file.length());
		}

		List<File> files = c.getSortedKeys();

		AtomicInteger file_cnt = new AtomicInteger(0);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(merge_thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(merge_thread_size);

		for (int i = 0; i < merge_thread_size; i++) {
			fs.add(tpe.submit(new MergeWorker(files, file_cnt, timer)));
		}

		for (int i = 0; i < merge_thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

	}

	private void mergePostingLists() throws Exception {
		System.out.println("merge posting lists.");

		Timer timer = Timer.newTimer();

		Vocab vocab = dc.getVocab();
		LongArray starts = new LongArray(vocab.size());
		IntegerArray lens = new IntegerArray(vocab.size());

		{
			for (int i = 0; i < vocab.size(); i++) {
				starts.add(-1);
				lens.add(0);
			}

			List<FileChannel> ins = Generics.newArrayList(output_file_size);

			for (File file : FileUtils.getFilesUnder(tmpDir)) {
				if (file.getName().startsWith("m_")) {
					ins.add(FileUtils.openFileChannel(file, "r"));
				}
			}

			File postingFile = new File(dataDir, DiskInvertedIndex.DATA_NAME);
			postingFile.delete();

			FileChannel out = FileUtils.openFileChannel(postingFile, "rw");

			for (int w = 0; w < vocab.size(); w++) {
				int file_loc = w % output_file_size;

				FileChannel fc = ins.get(file_loc);
				PostingList pl = PostingList.readPostingList(fc);

				if (pl == null) {
					continue;
				}

				if (pl.getWord() != w) {
					System.out.println();
					System.out.printf("w=%d, %s\n", w, pl.toString());
				}

				long[] info = PostingList.writePostingList(pl, out);

				starts.set(w, info[0]);
				lens.set(w, (int) info[1]);

				int prog = BatchUtils.progress(w + 1, vocab.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, w, vocab.size(), timer.stop());
				}

				// if (w % 10000 == 0 || w == vocab.size() - 1) {
				// System.out.printf("[%d percent, %d/%d, %s]\n", prog, w, vocab.size(), timer.stop());
				// }
			}
			out.close();

			for (FileChannel in : ins) {
				in.close();
			}

			FileUtils.deleteFilesUnder(tmpDir);
		}

		{
			File metaFile = new File(dataDir, DiskInvertedIndex.META_NAME);
			metaFile.delete();

			DataCompression.encodeGaps(starts);

			ByteArrayMatrix data = new ByteArrayMatrix(3);
			data.add(ByteArrayUtils.toByteArray(dc.size()));
			data.add(DataCompression.encode(starts));
			data.add(DataCompression.encode(lens));

			FileChannel fc = FileUtils.openFileChannel(metaFile, "rw");
			FileUtils.write(data, fc);
			fc.close();
		}
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setMergingThreadSize(int thread_size) {
		this.merge_thread_size = thread_size;
	}

	public void setPostingThreadSize(int thread_size) {
		this.posting_thread_size = thread_size;
	}

}
