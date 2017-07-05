package ohs.ir.search.app;

import java.io.File;
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
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
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
public class PhraseWeightEstimator {

	public enum Type {
		CDD, QDLM, IDF, DLM, LEN, STOP_RATIO
	}

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
		System.out.println("process begins.");
		List<String> phrss = null;

		// {
		// phrss = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_sorted.txt");
		// }

		// IntegerArray cnts = new IntegerArray();
		// IntegerArray docFreqs = new IntegerArray();
		//
		// {
		// phrss = Generics.newLinkedList();
		//
		// for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt")) {
		// String[] ps = line.split("\t");
		// phrss.add(ps[0]);
		// cnts.add(Integer.parseInt(ps[2]));
		// docFreqs.add(Integer.parseInt(ps[3]));
		// }
		// phrss = Generics.newArrayList(phrss);
		// }

		// DocumentSearcher ds = new DocumentSearcher(MIRPath.DATA_DIR + "merged/col/dc/", MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);

		int window_size = 10;

		// for (int i = 0; i < dc.size() && i < 1; i++) {

		int dseq = 1000;

		Pair<String, IntegerArray> p = dc.get(dseq);

		IntegerArrayMatrix d = DocumentCollection.toMultiSentences(p.getSecond());
		SparseVector dv = dc.getDocVector(dseq);
		DenseMatrix sims = new DenseMatrix(dv.size());
		SparseVector biases = dv.copy();
		biases.setAll(0);

		// biases.set(dc.getVocab().indexOf("cancer"), 1);
		// biases.set(dc.getVocab().indexOf("breast"), 1);
		// biases.set(dc.getVocab().indexOf("risk"), 1);
		// biases.set(dc.getVocab().indexOf("cohort"), 1);

		String text = DocumentCollection.toText(dc.getVocab(), d);

		for (int i = 0; i < d.size(); i++) {
			IntegerArray sent = d.get(i);

			if (i == 0) {
				for (int w : sent) {
					biases.set(w, 1);
				}
			}

			int pos = 0;

			for (int j = 1; j < sent.size(); j++) {
				int start = Math.max(j - window_size, 0);
				int w_center = sent.get(j);
				int loc1 = dv.location(w_center);
				String word1 = dc.getVocab().getObject(w_center);

				for (int k = start; k < j; k++) {
					int w_left = sent.get(k);
					int loc2 = dv.location(w_left);
					String word2 = dc.getVocab().getObject(w_left);

					double dist = (j - k);
					double cocnt = 1f / dist;
					double pos_decay = 1d / (pos + 1);

					cocnt *= pos_decay;

					sims.add(loc1, loc2, cocnt);
					sims.add(loc2, loc1, cocnt);
				}
			}
		}

		double prior_dir = 2500;
		double len_d = dv.sum();
		double mixture_jm = 0;

		for (int j = 0; j < biases.size(); j++) {
			int w = biases.indexAt(j);
			if (biases.valueAt(j) == 0) {
				continue;
			}

			double cnt_w_in_d = dv.valueAt(j);
			double pr_w_in_c = dc.getVocab().getProb(w);
			double pr_w_in_bg = pr_w_in_c;

			double weight = TermWeighting.tfidf(cnt_w_in_d, dc.size(), dc.getVocab().getDocFreq(w));
			// double weight = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_bg, mixture_jm);
			biases.setAt(j, weight);
		}
		biases.normalizeAfterSummation();

		for (int j = 0; j < sims.size(); j++) {
			int w1 = dv.indexAt(j);
			double cnt_w1_in_d = dv.valueAt(j);
			double pr_w1_in_c = dc.getVocab().getProb(w1);
			double pr_w1_in_bg = pr_w1_in_c;
			double weight1 = TermWeighting.tfidf(cnt_w1_in_d, dc.size(), dc.getVocab().getDocFreq(w1));
			// double weight1 = TermWeighting.twoStageSmoothing(cnt_w1_in_d, len_d, pr_w1_in_c, prior_dir, pr_w1_in_bg, mixture_jm);

			for (int k = j + 1; k < sims.size(); k++) {
				int w2 = dv.indexAt(k);
				double cnt_w2_in_d = dv.valueAt(k);
				double pr_w2_in_c = dc.getVocab().getProb(w2);
				double pr_w2_in_bg = pr_w2_in_c;
				double weight2 = TermWeighting.tfidf(cnt_w2_in_d, dc.size(), dc.getVocab().getDocFreq(w2));
				// double weight2 = TermWeighting.twoStageSmoothing(cnt_w2_in_d, len_d, pr_w2_in_c, prior_dir, pr_w2_in_bg, mixture_jm);
				double sim = sims.value(j, k);
				double new_sim = sim * weight1 * weight2;
				sims.set(j, k, new_sim);
				sims.set(k, j, new_sim);
			}
		}

		sims.normalizeColumns();

		SparseVector cents = dv.copy();
		cents.setAll(0);

		ArrayMath.randomWalk(sims.values(), cents.values(), biases.values(), 100);

		System.out.println(text);
		System.out.println(VectorUtils.toCounter(biases, dc.getVocab()).toStringSortedByValues(true, true, 20, "\t"));
		System.out.println(VectorUtils.toCounter(cents, dc.getVocab()).toStringSortedByValues(true, true, 50, "\t"));
		// }

		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
		// WordFilter wf = new WordFilter(dc.getVocab(), FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE));
		//
		// PhraseWeightEstimator dpe = new PhraseWeightEstimator(dc, wf, phrss);
		// dpe.setBatchSize(2000);
		// dpe.setThreadSize(10);
		// dpe.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		// docPriors.writeObject(dataDir + "doc_prior_quality.ser.gz");

		System.out.println("process ends.");
	}

	private int batch_size = 1000;

	private int thread_size = 5;

	private double prior_dir = 2000;

	private double mixture_jm = 0;

	private DocumentCollection dc;

	private Vocab vocab;

	private WordFilter wf;

	private Type type = Type.CDD;

	private SparseVector Q;

	private Indexer<String> phrsIndexer;

	public PhraseWeightEstimator(DocumentCollection dc, WordFilter wf, List<String> phrss) {
		this.dc = dc;
		this.vocab = dc.getVocab();
		this.wf = wf;

		phrsIndexer = Generics.newIndexer(phrss.size());

		for (String phrs : phrss) {
			phrsIndexer.add(phrs);
		}
	}

	public PhraseWeightEstimator(String idxDir) throws Exception {
		dc = new DocumentCollection(idxDir);
		vocab = dc.getVocab();
	}

	public DenseVector estimate(String outFileName) throws Exception {
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

	public double getDirichletPrior() {
		return prior_dir;
	}

	public double getMixtureJM() {
		return mixture_jm;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setDirichletPrior(double dirichlet_prior) {
		this.prior_dir = dirichlet_prior;
	}

	public void setJmMixture(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setQuery(SparseVector Q) {
		this.Q = Q;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	public void setType(Type type) {
		this.type = type;
	}

}
