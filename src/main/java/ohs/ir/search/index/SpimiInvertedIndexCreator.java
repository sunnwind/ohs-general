package ohs.ir.search.index;

import java.io.File;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DataCompression;
import ohs.corpus.type.DocumentCollection;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.neuralnet.com.BatchUtils;
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
public class SpimiInvertedIndexCreator {

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
				IntegerArrayMatrix subranges = new IntegerArrayMatrix(BatchUtils.getBatchRanges(batch_size, num_docs_in_file));
				File outDir = new File(tmpDir, new DecimalFormat(DIGIT_PATTERN).format(range.get(0) / num_files_in_dir));

				for (int m = 0; m < subranges.size(); m++) {
					IntegerArray subrange = subranges.get(m);
					subrange.set(0, range.get(0) + subrange.get(0));
					subrange.set(1, range.get(0) + subrange.get(1));

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
						File outFile = new File(outDir, new DecimalFormat(DIGIT_PATTERN).format(m) + ".ser");
						FileChannel fc = FileUtils.openFileChannel(outFile, "rw");
						write(fc, lmm);
						fc.close();
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
				long[] info = PostingList.writePostingList(pl, fc);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		SpimiInvertedIndexCreator iic = new SpimiInvertedIndexCreator();
		iic.setBatchSize(200);
		iic.setPostingThreadSize(1);
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

	public static void sort(PostingList pl) {
		Map<Integer, IntegerArray> m = Generics.newHashMap(pl.size());
		for (int i = 0; i < pl.size(); i++) {
			int dseq = pl.getDocSeqs().get(i);
			IntegerArray poss = pl.getPosData().get(i);
			m.put(dseq, poss);
		}

		pl.getDocSeqs().sort(false);

		for (int i = 0; i < pl.size(); i++) {
			int dseq = pl.getDocSeqs().get(i);
			pl.getPosData().set(i, m.get(dseq));
		}
	}

	private int num_files_in_dir = 100000;

	private int num_docs_in_file = 1000;

	private int posting_thread_size = 1;

	private String DIGIT_PATTERN = "0000000";

	private int batch_size = 500;

	private DocumentCollection dc;

	private File dataDir;

	private File tmpDir;

	public SpimiInvertedIndexCreator() {

	}

	public void check(IntegerArray a) {
		for (int i = 1; i < a.size(); i++) {
			int idx1 = a.get(i - 1);
			int idx2 = a.get(i);

			if (idx1 >= idx2) {
				System.out.println();
			}
		}
	}

	public void create(String dataDir) throws Exception {
		System.out.printf("create inverted index at [%s]\n", dataDir);

		this.dataDir = new File(dataDir);
		this.tmpDir = new File(dataDir, "tmp");
		dc = new DocumentCollection(dataDir);

		IntegerArray wordCnts = dc.getVocab().getCounts();
		IntegerArray docFreqs = dc.getVocab().getDocFreqs();

//		createPostingListFiles();
		mergePostingLists();
		//
		// test();
	}

	private void createPostingListFiles() throws Exception {
		System.out.println("create posting list files.");

		Timer timer = Timer.newTimer();

		FileUtils.deleteFilesUnder(tmpDir);

		tmpDir.mkdirs();

		IntegerArrayMatrix ranges = new IntegerArrayMatrix(BatchUtils.getBatchRanges(dc.size(), num_files_in_dir));

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

	private void mergePostingLists() throws Exception {
		System.out.println("merge posting lists.");

		Timer timer = Timer.newTimer();

		Vocab vocab = dc.getVocab();

		LinkedList<File> files = Generics.newLinkedList();

		for (File file : tmpDir.listFiles()) {
			if (file.isDirectory()) {
				if (file.getName().startsWith("m_")) {
					FileUtils.deleteFilesUnder(file);
				} else {
					for (File f : file.listFiles()) {
						files.addLast(f);
					}
				}
			}
		}

		Collections.sort(files);
		int file_cnt = 0;

		LongArray starts = new LongArray(new long[vocab.size()]);
		IntegerArray lens = new IntegerArray(new int[vocab.size()]);

		while (files.size() != 1) {
			File inFile1 = files.removeFirst();
			File inFile2 = files.removeFirst();
			File outDir = new File(tmpDir, String.format("m_%s", new DecimalFormat(DIGIT_PATTERN).format(file_cnt / num_docs_in_file)));
			File outFile = new File(outDir, new DecimalFormat(DIGIT_PATTERN).format(file_cnt) + ".ser");
			file_cnt++;

			FileChannel in1 = FileUtils.openFileChannel(inFile1, "r");
			FileChannel in2 = FileUtils.openFileChannel(inFile2, "r");
			FileChannel out = FileUtils.openFileChannel(outFile, "rw");

			PostingList pl1 = PostingList.readPostingList(in1);
			PostingList pl2 = PostingList.readPostingList(in2);
			int merge_cnt = 0;

			while (pl1 != null || pl2 != null) {
				int w = 0;
				long[] info = new long[2];

				// if (pl1.getWord() == 2 || pl1.getWord() == 4 || pl1.getWord() == 5) {
				// System.out.printf("pl1 + : %s\n", pl1);
				// }
				//
				// if (pl2.getWord() == 2 || pl2.getWord() == 4 || pl2.getWord() == 5) {
				// System.out.printf("pl1 + : %s\n", pl2);
				// }

				if (pl1 == null || pl2 != null && pl1.getWord() > pl2.getWord()) {
					w = pl2.getWord();
					check(pl2.getDocSeqs());

					info = PostingList.writePostingList(pl2, out);
					pl2 = PostingList.readPostingList(in2);
				} else if (pl2 == null || pl1.getWord() < pl2.getWord()) {
					w = pl1.getWord();
					check(pl1.getDocSeqs());

					info = PostingList.writePostingList(pl1, out);
					pl1 = PostingList.readPostingList(in1);
				} else {
					merge_cnt++;
					w = pl1.getWord();

					check(pl1.getDocSeqs());
					check(pl2.getDocSeqs());

					PostingList head = pl1;
					PostingList tail = pl2;

					int dseq1 = head.getDocSeqs().get(head.size() - 1);
					int dseq2 = tail.getDocSeqs().get(0);

					if (dseq1 >= dseq2) {
						head = pl2;
						tail = pl1;
					}

					head.getDocSeqs().addAll(tail.getDocSeqs());
					head.getPosData().addAll(tail.getPosData());

					Map<Integer, IntegerArray> m = Generics.newHashMap(head.size());
					for (int i = 0; i < head.size(); i++) {
						int dseq = head.getDocSeqs().get(i);
						IntegerArray poss = head.getPosData().get(i);
						m.put(dseq, poss);
					}

					head.getDocSeqs().sort(false);

					for (int i = 0; i < head.size(); i++) {
						int dseq = head.getDocSeqs().get(i);
						head.getPosData().set(i, m.get(dseq));
					}

					check(head.getDocSeqs());

					head = new PostingList(head.getWord(), head.getDocSeqs(), head.getPosData());
					// tmp1.getDocSeqs().trimToSize();
					// tmp1.getPosData().trimToSize();

					info = PostingList.writePostingList(head, out);

					pl1 = PostingList.readPostingList(in1);
					pl2 = PostingList.readPostingList(in2);
				}

				if (files.size() == 0) {
					starts.set(w, info[0]);
					lens.set(w, (int) info[1]);
				}
			}

			in1.close();
			in2.close();
			out.close();

			{
				File[] inFiles = new File[] { inFile1, inFile2 };
				for (File f : inFiles) {
					File parent = f.getParentFile();
					if (parent.getName().startsWith("m_")) {
						f.delete();
					}
					if (parent.listFiles().length == 0) {
						parent.delete();
					}
				}
			}

			files.add(outFile);

			if (file_cnt % 10 == 0) {
				System.out.printf("[%d, %d, %s]\n", file_cnt, files.size(), timer.stop());
			}
		}
		System.out.printf("[%d, %d, %s]\n", file_cnt, files.size(), timer.stop());

		{
			File dataFile = new File(dataDir, DiskInvertedIndex.DATA_NAME);
			dataFile.delete();

			files.get(0).renameTo(dataFile);
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

	public void setPostingThreadSize(int thread_size) {
		this.posting_thread_size = thread_size;
	}

	public void test() throws Exception {

		FileChannel fc = FileUtils.openFileChannel(new File(dataDir, DiskInvertedIndex.DATA_NAME), "r");

		PostingList pl = null;
		int cnt = 0;

		while ((pl = PostingList.readPostingList(fc)) != null) {
			if (++cnt > 100) {
				break;
			}

			IntegerArray dseqs = pl.getDocSeqs();

			System.out.println(pl.toString());
			System.out.println(dc.getVocab().getDocFreq(pl.getWord()));
			System.out.println();

			// for (int i = 0; i < dseqs.size() && i < 100; i++) {
			// int dseq = dseqs.get(i);
			// System.out.println(i + ", " + pl.toString());

			// if (dseq < 0 || dseq >= dc.size()) {
			// System.out.println(pl.toString());
			// System.out.println();
			// }
			// }
		}

		fc.close();
	}

}
