package ohs.corpus.search.app;

import java.io.File;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.lucene.queries.function.valuesource.MaxDocValueSource;

import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.Posting;
import ohs.corpus.search.index.PostingList;
import ohs.corpus.search.index.WordFilter;
import ohs.corpus.search.model.FeedbackBuilder;
import ohs.corpus.search.model.LMScorer;
import ohs.corpus.search.model.MRFScorer;
import ohs.corpus.search.model.Scorer;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class DocumentSearcher {

	class SearchWorker implements Callable<Map<Integer, SparseVector>> {

		private List<SparseVector> queryData;

		private List<SparseVector> scoreData;

		private DocumentSearcher ds;

		private AtomicInteger q_cnt;

		public SearchWorker(DocumentSearcher ds, List<SparseVector> queryData, List<SparseVector> scoreData, AtomicInteger q_cnt) {
			this.ds = ds;
			this.queryData = queryData;
			this.scoreData = scoreData;
			this.q_cnt = q_cnt;
		}

		@Override
		public Map<Integer, SparseVector> call() throws Exception {
			int q_loc = 0;
			Map<Integer, SparseVector> ret = Generics.newHashMap();

			while ((q_loc = q_cnt.getAndIncrement()) < queryData.size()) {
				SparseVector Q = queryData.get(q_loc);
				SparseVector scores = null;

				StringBuffer sb = new StringBuffer();
				sb.append("-------------------------");
				sb.append(String.format("\nthread name:\t%s", Thread.currentThread().getName()));
				sb.append("\n" + StrUtils.join(" ", ds.getVocab().getObjects(Q.indexes())));
				sb.append("\n" + VectorUtils.toCounter(Q, ds.getVocab()));
				System.out.println(sb.toString() + "\n\n");

				if (scoreData == null) {
					scores = ds.search(Q);
				} else {
					SparseVector prevScores = scoreData.get(q_loc);
					prevScores.sortIndexes();
					scores = ds.search(Q, prevScores);
					prevScores.sortValues();
				}
				ret.put(q_loc, scores);
			}
			return ret;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			rdc.close();
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		test1();
		// test2();

		System.out.println("process ends.");
	}

	public static void test1() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.CLUEWEB_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentSearcher ds = new DocumentSearcher(MIRPath.OHSUMED_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		ds.setScorer(new MRFScorer(ds));

		String Qstr = "lung cancer treatment";
		// String Qstr = "( )";
		SparseVector Q = ds.index(Qstr);

		SparseVector scores = ds.search(Q);

		System.out.println("Q:");
		System.out.println(StrUtils.join(" ", ds.getVocab().getObjects(Q.indexes())));
		System.out.println(scores.size());

		for (int i = 0; i < scores.size() && i < 10; i++) {
			int dseq = scores.indexAt(i);
			double score = scores.valueAt(i);

			Pair<String, IntegerArrayMatrix> p = ds.getDocumentCollection().getSents(dseq);
			String did = p.getFirst();
			IntegerArrayMatrix doc = p.getSecond();
			SparseVector dv = ds.getDocumentCollection().getDocVector(i);
			Counter<Integer> c = Generics.newCounter();

			for (int w : Q.indexes()) {
				c.setCount(w, dv.value(w));
			}

			IntegerArrayMatrix subdoc = doc.subMatrix(0, Math.min(doc.size(), 5));

			String text = DocumentCollection.getText(ds.getVocab(), subdoc);

			System.out.println("======== Document ======");
			System.out.printf("docid: %s, dseq: %d, score: %f\n", did, dseq, score);
			System.out.printf("cnts: %s\n", VectorUtils.toCounter(c, ds.getDocumentCollection().getVocab()));
			System.out.println(text);
			System.out.println();
		}
	}

	public static void test2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.CLUEWEB_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		InvertedIndex ii = ds.getInvertedIndex();

		String Qstr = "lung cancer treatment";
		SparseVector Q = ds.index(Qstr);

		long len1 = 0;
		long len2 = 0;

		for (int i = 0; i < ii.size() && i < 100000; i++) {
			String word = ds.getVocab().getObject(i);
			PostingList pl = ii.getPostingList(i);

			for (int j = 0; j < pl.size(); j++) {
				Posting p = pl.getPosting(j);
				IntegerArray poss = p.getPoss();

				long l1 = poss.size() * Integer.BYTES;

				BitSet bs = new BitSet();

				for (int pos : poss) {
					bs.set(pos);
				}
				long l2 = bs.toLongArray().length * Long.BYTES;

				len1 += l1;
				len2 += l2;

				if (l2 < l1) {
					double ratio = 1f * l2 / l1;
					System.out.printf("[%s, %d, %d, %f], %s\n", word, l1, l2, ratio, pl);
				}
			}
		}
		double ratio = len2 / len1;

		System.out.printf("[%d, %d, %f]\n", len1, len2, ratio);

		// int[] a = ArrayUtils.range(10000);
		// // ArrayUtils.shuffle(a);
		//
		// int[] b = ArrayUtils.copy(a, 0, 100);
		//
		// BitSet bs = new BitSet();
		//
		// for (int c : b) {
		// bs.set(c);
		// }
		//
		// long len1 = b.length * Integer.BYTES;
		// long len2 = bs.toByteArray().length;
		// long len3 = bs.size();
		// double ratio = 1f * len2 / len1;
		//
		// System.out.printf("[%d, %d, %d, %f]\n", len1, len2, len3, ratio);

	}

	private StringNormalizer sn = new SimpleStringNormalizer(true);

	private Vocab vocab;

	private InvertedIndex ii;

	private DocumentCollection dc;

	private RawDocumentCollection rdc;

	private Scorer scorer;

	private File dataDir;

	private WordFilter wf;

	private double mixture_fb = 0.5;

	private int max_match_size = Integer.MAX_VALUE;

	private int top_k = Integer.MAX_VALUE;

	private boolean use_idf_match = false;

	private boolean print_log = false;

	private LemmaExpander le;

	public DocumentSearcher(Scorer scorer, RawDocumentCollection rdc, DocumentCollection dc, InvertedIndex ii, WordFilter wf)
			throws Exception {
		this.scorer = scorer;
		this.rdc = rdc;
		this.dc = dc;
		this.ii = ii;
		this.wf = wf;
		this.vocab = dc.getVocab();
	}

	public DocumentSearcher(String dataDir, String stopwordFileName) throws Exception {
		this.dataDir = new File(dataDir);

		rdc = new RawDocumentCollection(dataDir);

		dc = new DocumentCollection(dataDir);
		vocab = dc.getVocab();

		ii = new InvertedIndex(dataDir);
		ii.setVocab(vocab);

		scorer = new LMScorer(vocab, dc, ii);

		if (stopwordFileName != null) {
			Set<String> stopwords = FileUtils.readStringSetFromText(stopwordFileName);
			wf = new WordFilter(vocab, stopwords);
		} else {
			wf = new WordFilter(vocab, null);
		}
	}

	public void close() throws Exception {
		ii.close();
		rdc.close();
		dc.close();
	}

	public DocumentSearcher copyShallow() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(scorer, rdc.copyShallow(), dc.copyShallow(), ii.copyShallow(), wf);
		ds.setFeedbackMixture(mixture_fb);
		ds.setMaxMatchSize(max_match_size);
		ds.setTopK(top_k);
		ds.setUseIdfMatch(use_idf_match);
		return ds;
	}

	public File getDataDir() {
		return dataDir;
	}

	public DocumentCollection getDocumentCollection() {
		return dc;
	}

	public String getDocumentText(IntegerArray Q, int docseq) throws Exception {
		Pair<String, IntegerArray> p = dc.get(docseq);
		IntegerArrayMatrix doc = DocumentCollection.toMultiSentences(p.getSecond());

		StringBuffer sb = new StringBuffer();
		sb.append("<-");
		sb.append(String.format("\ndocseq:\t%d", docseq));
		sb.append(String.format("\ndocid:\t%s", p.getFirst()));
		if (Q == null) {
			sb.append("\n" + DocumentCollection.getText(vocab, doc));
		} else {
			for (int i = 0; i < doc.size(); i++) {
				IntegerArray sent = doc.get(i);
				Set<Integer> ws = Generics.newHashSet();
				for (int j = 0; j < sent.size(); j++) {
					ws.add(sent.get(j));
				}

				boolean has_q = false;

				for (int q : Q) {
					if (ws.contains(q)) {
						has_q = true;
						break;
					}
				}

				if (has_q) {
					sb.append(String.format("\n%d:\t%s", i, StrUtils.join(" ", vocab.getObjects(sent.values()))));
				}
			}
		}

		sb.append("\n->");
		return sb.toString();
	}

	public double getFeedbackMixture() {
		return mixture_fb;
	}

	public InvertedIndex getInvertedIndex() {
		return ii;
	}

	public RawDocumentCollection getRawDocumentCollection() {
		return rdc;
	}

	public Scorer getScorer() {
		return scorer;
	}

	public StringNormalizer getStringNormalizer() {
		return sn;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public WordFilter getWordFilter() {
		return wf;
	}

	public SparseVector index(String Q) {
		return index(Q, true);
	}

	public SparseVector index(String Q, boolean keep_order) {
		Q = sn.normalize(Q);

		IntegerArray ws = new IntegerArray();

		for (String word : StrUtils.split(Q)) {
			int w = vocab.indexOf(word);
			if (wf.filter(w)) {
				continue;
			}
			ws.add(w);
		}

		SparseVector ret = new SparseVector(0);

		if (keep_order) {
			ret = new SparseVector(ws.size());
			for (int i = 0; i < ws.size(); i++) {
				ret.addAt(i, ws.get(i), 1);
			}
		} else {
			Counter<Integer> c = Generics.newCounter();
			for (int w : ws) {
				c.incrementCount(w, 1);
			}
			ret = new SparseVector(c);
		}

		return ret;
	}

	public SparseVector match(SparseVector Q) throws Exception {
		Timer timer = Timer.newTimer();

		int max_doc_freq = 0;
		for (int w : Q.indexes()) {
			int doc_freq = vocab.getDocFreq(w);
			max_doc_freq = Math.max(doc_freq, max_doc_freq);
		}

		Counter<Integer> c = Generics.newCounter(max_doc_freq);

		for (int w : Q.indexes()) {
			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}

			IntegerArray dseqs = pl.getDocSeqs();
			double doc_freq = dseqs.size();
			double idf = TermWeighting.idf(vocab.getDocCnt(), doc_freq);

			for (int dseq : dseqs) {
				double val = use_idf_match ? idf : 1;
				c.incrementCount(dseq, val);
			}
		}

		SparseVector ret = new SparseVector(c);

		if (max_match_size < Integer.MAX_VALUE) {
			ret.sortValues();
			ret = ret.subVector(max_match_size);
			ret.sortIndexes();
		}

		System.out.printf("[Matching Time, %s]\n", timer.stop());
		return ret;
	}

	public SparseVector matchSeq(SparseVector Q, boolean keep_order, int window_size) throws Exception {
		Timer timer = Timer.newTimer();

		PostingList pl = matchSeqPostingList(Q, keep_order, window_size);
		SparseVector ret = new SparseVector(0);
		if (pl != null) {
			IntegerArray dseqs = pl.getDocSeqs();
			IntegerArray cnts = pl.getCounts();
			ret = new SparseVector(dseqs.size());

			for (int i = 0; i < dseqs.size(); i++) {
				ret.addAt(i, dseqs.get(i), cnts.get(i));
			}

			if (max_match_size < Integer.MAX_VALUE) {
				ret.sortValues();
				ret = ret.subVector(max_match_size);
			}
			ret.sortIndexes();
		}

		System.out.printf("[Matching Time, %s]\n", timer.stop());
		return ret;
	}

	public PostingList matchSeqPostingList(SparseVector Q, boolean keep_order, int window_size) throws Exception {
		return ii.getPostingList(new IntegerArray(Q.indexes()), keep_order, window_size);
	}

	public List<SparseVector> search(List<SparseVector> qData, List<SparseVector> dData, int thread_size) throws Exception {

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Map<Integer, SparseVector>>> fs = Generics.newArrayList(thread_size);

		List<DocumentSearcher> dss = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			dss.add(copyShallow());
		}

		AtomicInteger q_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new SearchWorker(dss.get(i), qData, dData, q_cnt)));
		}

		List<SparseVector> ret = Generics.newArrayList(qData.size());

		for (int i = 0; i < qData.size(); i++) {
			ret.add(new SparseVector(0));
		}

		for (int i = 0; i < thread_size; i++) {
			Map<Integer, SparseVector> res = fs.get(i).get();
			for (Entry<Integer, SparseVector> e : res.entrySet()) {
				ret.set(e.getKey(), e.getValue());
			}
		}
		fs.clear();

		tpe.shutdown();

		for (int i = 0; i < thread_size; i++) {
			dss.get(i).close();
		}

		return ret;
	}

	public SparseVector search(SparseVector Q) throws Exception {
		SparseVector ret = null;

		if (le == null) {
			ret = search(Q, match(Q));
		} else {

		}

		return ret;
	}

	public SparseVector search(SparseVector Q, SparseVector docs) throws Exception {
		Timer timer = Timer.newTimer();
		SparseVector ret = scorer.score(Q, docs);
		scorer.postprocess(ret);

		if (top_k != Integer.MAX_VALUE) {
			ret = ret.subVector(top_k);
		}

		System.out.printf("[Scoring Time, %s]\n", timer.stop());
		return ret;
	}

	public SparseVector search(String Q) throws Exception {
		return search(index(Q));
	}

	public void setFeedbackMixture(double mixture_fb) {
		this.mixture_fb = mixture_fb;
	}

	public void setLemmaExpander(LemmaExpander le) {
		this.le = le;
	}

	public void setMaxMatchSize(int max_match_size) {
		this.max_match_size = max_match_size;
	}

	public void setPrintLog(boolean print_log) {
		this.print_log = print_log;
	}

	public void setScorer(Scorer scorer) {
		this.scorer = scorer;
	}

	public void setStringNormalizer(StringNormalizer sn) {
		this.sn = sn;
	}

	public void setTopK(int top_k) {
		this.top_k = top_k;
	}

	public void setUseIdfMatch(boolean use_idf_match) {
		this.use_idf_match = use_idf_match;
	}

}
