package ohs.corpus.type;

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

import ohs.io.ByteArray;
import ohs.io.ByteArrayMatrix;
import ohs.io.ByteArrayUtils;
import ohs.io.ByteBufferWrapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
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

public class DocumentIdMapCreator {

	public class Worker implements Callable<Integer> {

		private int[][] ranges;

		private AtomicInteger range_cnt;

		private DocumentCollection dc;

		private List<String> docids;

		private AtomicInteger doc_cnt;

		private Timer timer;

		public Worker(DocumentCollection dc, List<String> docids, int[][] ranges, AtomicInteger range_cnt, AtomicInteger doc_cnt,
				Timer timer) {
			this.dc = dc;
			this.docids = docids;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.doc_cnt = doc_cnt;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int range_loc = 0;
			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] range = ranges[range_loc];
				List<Pair<String, IntegerArray>> ps = dc.getRange(range);
				for (int i = 0; i < ps.size(); i++) {
					Pair<String, IntegerArray> p = ps.get(i);
					int dseq = range[0] + i;
					docids.set(dseq, p.getFirst());

					int cnt = doc_cnt.incrementAndGet();

					int prog = BatchUtils.progress(cnt, dc.size());

					if (prog > 0) {
						System.out.printf("[%s percent, %d/%d, %s]\n", prog, cnt, dc.size(), timer.stop());
					}
				}
			}
			return null;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			dc.close();
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentIdMapCreator dic = new DocumentIdMapCreator();
		dic.setThreadSize(5);

		dic.create(MIRPath.CLUEWEB_COL_DC_DIR);

		// dic.create(MIRPath.OHSUMED_COL_DC_DIR, 1, new int[] { 3, 5 });
		// dic.create(MIRPath.TREC_CDS_2014_COL_DC_DIR, 0, new int[] { 1, 2, 3 });
		// dic.create(MIRPath.TREC_CDS_2016_COL_DC_DIR, 0, new int[] { 1, 2, 3 });
		// dic.create("../../data/medical_ir/scopus/col/dc/", 0, new int[] { 1, 2 });
		// dic.create(MIRPath.BIOASQ_COL_DC_DIR, 0, new int[] { 4, 5 });
		// dic.create(MIRPath.WIKI_COL_DC_DIR, 0, new int[] { 3 });
		// dic.create(MIRPath.CLUEWEB_COL_DC_DIR, 0, new int[] { 1 });

		// dic.create(MIRPath.CLEF_EH_2014_COL_DC_DIR, 0, new int[] { 3 });
		// dic.create(MIRPath.TREC_GENO_2007_COL_DC_DIR, 0, new int[] { 1 });
		// dic.create(MIRPath.CLUEWEB_COL_DC_DIR, 0, new int[] { 1 });

		// dic.setStringNormalizer(new ThreePStringNormalizer());
		// dic.create(KPPath.COL_DC_DIR, 0, new int[] { 4, 5, 6, 7 });

		System.out.println("process ends.");
	}

	private boolean encode = true;

	private int thread_size = 1;

	private File dataDir;

	private DocumentCollection dc;

	public DocumentIdMapCreator() {

	}

	public void create(String dataDir) throws Exception {
		System.out.printf("create document id map at [%s]\n", dataDir);
		this.dataDir = new File(dataDir);

		dc = new DocumentCollection(dataDir);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		int[][] ranges = BatchUtils.getBatchRanges(dc.size(), 500);

		AtomicInteger range_cnt = new AtomicInteger(0);

		AtomicInteger doc_cnt = new AtomicInteger(0);

		Timer timer = Timer.newTimer();

		List<String> docids = Generics.newArrayList(dc.size());

		for (int i = 0; i < dc.size(); i++) {
			docids.add("");
		}

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(dc.copyShallow(), docids, ranges, range_cnt, doc_cnt, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		tpe.shutdown();

		write(docids);

	}

	public void setEncode(boolean encode) {
		this.encode = encode;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	private void write(List<String> dids) throws Exception {
		System.out.println("write doc-id map.");
		Timer timer = Timer.newTimer();

		File outFile1 = new File(dataDir, DocumentIdMap.DATA_NAME);
		File outFile2 = new File(dataDir, DocumentIdMap.META_NAME);

		FileUtils.delete(outFile1.getPath());
		FileUtils.delete(outFile2.getPath());

		FileChannel out1 = FileUtils.openFileChannel(outFile1, "rw");
		FileChannel out2 = FileUtils.openFileChannel(outFile2, "rw");

		LongArray starts = new LongArray(dids.size());
		IntegerArray lens = new IntegerArray(dids.size());

		for (int i = 0; i < dids.size(); i++) {
			String did = dids.get(i);
			ByteArray data = encode ? DataCompression.encode(did) : new ByteArray(did.getBytes());

			long[] info = FileUtils.write(data, out1);
			starts.add(info[0]);
			lens.add((int) info[1]);

			int prog = BatchUtils.progress(i + 1, dc.size());

			if (prog > 0) {
				System.out.printf("[%s percent, %d/%d, %s]\n", prog, i + 1, dc.size(), timer.stop());
			}
		}

		DataCompression.encodeGaps(starts);

		ByteArrayMatrix data = new ByteArrayMatrix();
		data.add(DataCompression.encode(starts));
		data.add(DataCompression.encode(lens));
		data.add(new ByteArray(new byte[] { (byte) (encode ? 1 : 0) }));

		FileUtils.write(data, out2);

		out1.close();
		out2.close();
	}

}
