package ohs.corpus.search.app;

import java.io.File;
import java.util.BitSet;
import java.util.Set;

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
import ohs.ir.weight.TermWeighting;
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

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		test1();
		// test2();

		System.out.println("process ends.");
	}

	public static void test2() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

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

	public static void test1() throws Exception {
		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentSearcher ds = new DocumentSearcher(MIRPath.OHSUMED_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		// String Qstr = "lung cancer treatment";
		String Qstr = "( )";
		SparseVector Q = ds.index(Qstr);

		{
			IntegerArray W = new IntegerArray();

			for (String w : Qstr.split(" ")) {
				W.add(ds.getVocab().indexOf(w));
			}

			Q = new SparseVector(W.size());

			for (int i = 0; i < W.size(); i++) {
				Q.addAt(i, W.get(i), 1);
			}
		}

		Set<Integer> toKeep = Generics.newHashSet();

		for (int w : Q.indexes()) {
			toKeep.add(w);
		}

		ds.setScorer(new MRFScorer(ds));

		// SPostingList pl = ds.matchSeqPostingList(new
		// IntegerArray(Q.indexes()), true, 5);

		// SparseVector scores = ds.matchSeq(new IntegerArray(Q.indexes()),
		// true, 5);
		// scores.sortValues();

		SparseVector scores = ds.search(Q);

		System.out.println("Q:");
		System.out.println(StrUtils.join(" ", ds.getVocab().getObjects(Q.indexes())));
		System.out.println(scores.size());

		for (int i = 0; i < scores.size() && i < 10; i++) {
			int dseq = scores.indexAt(i);
			double score = scores.valueAt(i);

			Pair<String, IntegerArrayMatrix> p = ds.getDocumentCollection().getSents(dseq);
			IntegerArrayMatrix doc = p.getSecond();

			Counter<Integer> c = Generics.newCounter();

			for (IntegerArray sent : doc) {
				for (int w : sent) {
					c.incrementCount(w, 1);
				}
			}

			c.pruneExcept(toKeep);

			IntegerArrayMatrix subdoc = doc.subMatrix(0, Math.min(doc.size(), 5));

			String text = DocumentCollection.getText(ds.getVocab(), subdoc);

			System.out.println("======== Document ======");
			System.out.printf("dseq: %d, score: %f\n", dseq, score);
			System.out.printf("cnts: %s\n", VectorUtils.toCounter(c, ds.getDocumentCollection().getVocab()));
			System.out.println(text);
			System.out.println();
		}
	}

	private StringNormalizer sn = new SimpleStringNormalizer(true);

	private Vocab vocab;

	private InvertedIndex ii;

	private DocumentCollection dc;

	private RawDocumentCollection rdc;

	private Scorer scorer;

	private File dataDir;

	public SparseVector Q;

	private FeedbackBuilder fbb;

	private WordFilter filter;

	private double mixture_fb = 0.5;

	private boolean print_log = false;

	private int max_match_size = Integer.MAX_VALUE;

	private boolean use_idf_match = false;

	public DocumentSearcher(Scorer scorer, RawDocumentCollection rdc, DocumentCollection ldc, InvertedIndex ii, WordFilter filter)
			throws Exception {
		this.scorer = scorer;
		this.rdc = rdc;
		this.dc = ldc;
		this.ii = ii;
		this.filter = filter;
	}

	public DocumentSearcher(String dataDir, String stopwordFileName) throws Exception {
		this.dataDir = new File(dataDir);

		dc = new DocumentCollection(dataDir);

		rdc = new RawDocumentCollection(dataDir);

		ii = new InvertedIndex(dataDir);

		vocab = dc.getVocab();

		scorer = new LMScorer(vocab, dc, ii);

		if (stopwordFileName != null) {
			Set<String> stopwords = FileUtils.readStringSetFromText(stopwordFileName);
			filter = new WordFilter(vocab, stopwords);
		} else {
			filter = new WordFilter(vocab, null);
		}

		fbb = new FeedbackBuilder(vocab, dc, ii, filter);
	}

	public void close() throws Exception {
		ii.close();
		rdc.close();
		dc.close();
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

	public FeedbackBuilder getFeedbackBuilder() {
		return fbb;
	}

	public InvertedIndex getInvertedIndex() {
		return ii;
	}

	public SparseVector getQueryModel(SparseVector Q, boolean use_feedback) throws Exception {
		SparseVector lm_q = VectorUtils.toSparseVector(Q);
		lm_q.normalize();

		if (use_feedback) {
			SparseVector scores = scorer.score(lm_q, match(new IntegerArray(lm_q.indexes())));
			SparseVector lm_fb = fbb.buildRM1(scores);
			SparseVector lm_q2 = updateQueryModel(lm_q, lm_fb);
			lm_q = lm_q2;
		} else {
			if (print_log) {
				StringBuffer sb = new StringBuffer();
				sb.append("<QLM>");
				sb.append(String.format("\nlm_q\t%s", VectorUtils.toCounter(lm_q, vocab)));
				System.out.println(sb.toString() + "\n");
			}
		}

		return lm_q;
	}

	public RawDocumentCollection getRawDocumentCollection() {
		return rdc;
	}

	public Scorer getScorer() {
		return scorer;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public WordFilter getWordFilter() {
		return filter;
	}

	public SparseVector index(String Q) {
		return index(Q, true);
	}

	public SparseVector index(String Q, boolean keep_order) {
		Q = sn.normalize(Q);

		IntegerArray ws = new IntegerArray();

		for (String word : Q.split("[\\s]+")) {
			int w = vocab.indexOf(word);
			if (filter.filter(w)) {
				continue;
			}
			ws.add(w);
		}

		SparseVector ret = null;

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

	public SparseVector match(IntegerArray Q) throws Exception {
		Timer timer = Timer.newTimer();

		{
			Set<Integer> set = Generics.newHashSet(Q.size());
			for (int w : Q) {
				set.add(w);
			}
			Q = new IntegerArray(set);
			Q.sort(false);
		}

		int max_docs = 0;
		for (int w : Q) {
			max_docs = Math.max(vocab.getDocFreq(w), max_docs);
		}
		Counter<Integer> c = Generics.newCounter(max_docs);

		for (int w : Q) {
			String word = vocab.getObject(w);
			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}

			IntegerArray dseqs = pl.getDocSeqs();
			double doc_freq = dseqs.size();
			double idf = TermWeighting.idf(vocab.getDocCnt(), doc_freq);

			for (int dseq : dseqs) {
				if (use_idf_match) {
					c.incrementCount(dseq, idf);
				} else {
					c.incrementCount(dseq, 1);
				}
			}
		}

		SparseVector ret = new SparseVector(c);
		if (max_match_size < Integer.MAX_VALUE) {
			ret.sortValues();
			ret = ret.subVector(max_match_size);
		}
		ret.sortIndexes();

		System.out.printf("[matching time, %s]\n", timer.stop());
		return ret;
	}

	public SparseVector matchSeq(IntegerArray Q, boolean keep_order, int window_size) throws Exception {
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

		System.out.printf("[matching time, %s]\n", timer.stop());
		return ret;
	}

	public PostingList matchSeqPostingList(IntegerArray Q, boolean keep_order, int window_size) throws Exception {
		return ii.getPostingList(Q, keep_order, window_size);
	}

	public SparseVector search(SparseVector Q) throws Exception {
		return search(Q, match(new IntegerArray(Q.indexes())));
	}

	public SparseVector search(SparseVector Q, boolean use_feedback) throws Exception {
		return search(getQueryModel(Q, use_feedback), match(new IntegerArray(Q.indexes())));
	}

	public SparseVector search(SparseVector Q, SparseVector docs) throws Exception {
		Timer timer = Timer.newTimer();
		SparseVector ret = scorer.score(Q, docs);
		scorer.postprocess(ret);
		System.out.printf("[scoring time, %s]\n", timer.stop());
		return ret;
	}

	public SparseVector search(String Q, boolean use_feedback) throws Exception {
		return search(index(Q), use_feedback);
	}

	public void setFeedbackMixture(double mixture_fb) {
		this.mixture_fb = mixture_fb;
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

	public void setUseIdfMatch(boolean use_idf_match) {
		this.use_idf_match = use_idf_match;
	}

	public SparseVector updateQueryModel(SparseVector lm_q, SparseVector lm_fb) {
		SparseVector lm_q2 = VectorMath.addAfterMultiply(lm_q, 1 - mixture_fb, lm_fb, mixture_fb);

		if (print_log) {
			StringBuffer sb = new StringBuffer();
			sb.append("<EXPAND>");
			sb.append(String.format("\nlm_q\t%s", VectorUtils.toCounter(lm_q, vocab)));
			sb.append(String.format("\nlm_fb\t%s", VectorUtils.toCounter(lm_fb, vocab)));
			sb.append(String.format("\nlm_q2\t%s\n", VectorUtils.toCounter(lm_q2, vocab)));
			System.out.println(sb.toString());
		}

		return lm_q2;
	}

}
