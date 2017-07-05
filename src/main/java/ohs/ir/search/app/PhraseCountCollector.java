package ohs.ir.search.app;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class PhraseCountCollector {

	class Worker implements Callable<Integer> {

		private AtomicInteger range_cnt;

		private IntegerArrayMatrix ranges;

		private DocumentCollection dc;

		private Timer timer;

		private Indexer<String> phrsIndexer;

		private PhraseMapper<Integer> pm;

		private Counter<Integer> phrsCnts;

		private Counter<Integer> phrsFreqs;

		public Worker(DocumentCollection dc, IntegerArrayMatrix ranges, AtomicInteger range_cnt, Indexer<String> phrsIndexer,
				PhraseMapper<Integer> pm, Counter<Integer> phrsCnts, Counter<Integer> phrsFreqs, Timer timer) {
			this.dc = dc;
			this.ranges = ranges;
			this.range_cnt = range_cnt;
			this.phrsIndexer = phrsIndexer;
			this.pm = pm;
			this.phrsCnts = phrsCnts;
			this.phrsFreqs = phrsFreqs;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int loc = 0;

			while ((loc = range_cnt.getAndIncrement()) < ranges.size()) {
				IntegerArray range = ranges.get(loc);

				List<Pair<String, IntegerArray>> res = dc.getRange(range.values());

				Counter<Integer> c1 = Generics.newCounter();
				Counter<Integer> c2 = Generics.newCounter();

				for (int i = 0; i < res.size(); i++) {
					int dseq = range.get(0) + i;
					IntegerArray d = res.get(i).getSecond();
					List<IntPair> ps = pm.map(d.toArrayList());

					Counter<Integer> tmp = Generics.newCounter();

					for (IntPair p : ps) {
						IntegerArray d_sub = d.subArray(p.getFirst(), p.getSecond());
						String phrs = StrUtils.join(" ", vocab.getObjects(d_sub));
						int pid = phrsIndexer.indexOf(phrs);
						tmp.incrementCount(pid, 1);
					}

					c1.incrementAll(tmp);
					c2.incrementAll(tmp.keySet(), 1);
				}

				synchronized (phrsCnts) {
					phrsCnts.incrementAll(c1);
				}

				synchronized (phrsFreqs) {
					phrsFreqs.incrementAll(c2);
				}

				int prog = BatchUtils.progress(loc, ranges.size());
				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %d/%d, %s]\n", prog, loc, ranges.size(), range.get(1), dc.size(), timer.stop());
				}
			}

			dc.close();

			return null;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			dc.close();
		}
	}

	public static void main(String[] args) throws Exception {
		List<String> phrss = null;

		// {
		// phrss = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_sorted.txt");
		// }

		{
			phrss = Generics.newLinkedList();

			for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_merged_2.txt")) {
				String[] ps = line.split("\t");
				int rsc_cnt = Integer.parseInt(ps[1]);
				if (rsc_cnt < 2) {
					continue;
				}
				phrss.add(ps[0]);
			}
			phrss = Generics.newArrayList(phrss);
		}
		// DocumentSearcher ds = new DocumentSearcher(MIRPath.DATA_DIR + "merged/col/dc/", MIRPath.STOPWORD_INQUERY_FILE);

		DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
		WordFilter wf = new WordFilter(dc.getVocab(), FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE));

		PhraseCountCollector dpe = new PhraseCountCollector(dc, wf, phrss);
		dpe.setBatchSize(2000);
		dpe.setThreadSize(10);
		dpe.collect(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		// docPriors.writeObject(dataDir + "doc_prior_quality.ser.gz");
	}

	private int batch_size = 1000;

	private int thread_size = 5;

	private DocumentCollection dc;

	private Vocab vocab;

	private WordFilter wf;

	private Indexer<String> phrsIndexer;

	public PhraseCountCollector(DocumentCollection dc, WordFilter wf, List<String> phrss) {
		this.dc = dc;
		this.vocab = dc.getVocab();
		this.wf = wf;

		phrsIndexer = Generics.newIndexer(phrss.size());

		for (String phrs : phrss) {
			phrsIndexer.add(phrs);
		}
	}

	public PhraseCountCollector(String idxDir) throws Exception {
		dc = new DocumentCollection(idxDir);
		vocab = dc.getVocab();
	}

	public DenseVector collect(String outFileName) throws Exception {
		Timer timer = Timer.newTimer();

		IntegerArrayMatrix ranges = new IntegerArrayMatrix(BatchUtils.getBatchRanges(dc.size(), batch_size));

		AtomicInteger range_cnt = new AtomicInteger(0);

		Counter<Integer> phrsCnts = Generics.newCounter(phrsIndexer.size());
		Counter<Integer> phrsFreqs = Generics.newCounter(phrsIndexer.size());

		PhraseMapper<Integer> pm = new PhraseMapper<Integer>(PhraseMapper.createTrie(phrsIndexer.getObjects(), vocab));

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(dc.copyShallow(), ranges, range_cnt, phrsIndexer, pm, phrsCnts, phrsFreqs, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		tpe.shutdown();

		List<String> res = Generics.newArrayList(phrsCnts.size());

		Counter<Integer> phrsWeights = Generics.newCounter(phrsCnts.size());

		for (int pid : phrsCnts.keySet()) {
			double tf = phrsCnts.getCount(pid);
			double doc_freq = phrsFreqs.getCount(pid);
			double tfidf = TermWeighting.tfidf(tf, dc.size(), doc_freq);
			phrsWeights.setCount(pid, tfidf);
		}

		for (int pid : phrsWeights.getSortedKeys()) {
			int phrs_cnt = (int) phrsCnts.getCount(pid);
			int phrs_freq = (int) phrsFreqs.getCount(pid);
			double weight = phrsWeights.getCount(pid);
			String phrs = phrsIndexer.getObject(pid);
			res.add(String.format("%s\t%f\t%d\t%d", phrs, weight, phrs_cnt, phrs_freq));
		}

		FileUtils.writeStringCollectionAsText(outFileName, res);

		return null;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
