package ohs.ir.search.app;

import java.util.List;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class KeyphraseExtractorOld {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsWeights = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt");
			phrsWeights = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				double weight = Double.parseDouble(ps[1]);
				phrsWeights.setCount(ps[0], weight);
			}
		}

		// DocumentSearcher ds = new DocumentSearcher(MIRPath.DATA_DIR + "merged/col/dc/", MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);

		int dseq = 1000;

		Pair<String, IntegerArray> p = dc.get(dseq);

		IntegerArrayMatrix d = DocumentCollection.toMultiSentences(p.getSecond());

		String text = DocumentCollection.toText(dc.getVocab(), d);
		System.out.println(text);

		KeyphraseExtractorOld ke = new KeyphraseExtractorOld(dc, null, phrsWeights);
		ke.extract(d);

		// biases.set(dc.getVocab().indexOf("cancer"), 1);
		// biases.set(dc.getVocab().indexOf("breast"), 1);
		// biases.set(dc.getVocab().indexOf("risk"), 1);
		// biases.set(dc.getVocab().indexOf("cohort"), 1);

		System.out.println("process ends.");
	}

	private Indexer<String> phrsIdxer;

	private PhraseMapper<Integer> pm;

	private DenseVector topicBiases;

	private SparseVector biases;

	private SparseVector dv;

	private IntegerArrayMatrix d;

	private DenseMatrix T;

	private int window_size = 10;

	private DocumentCollection dc;

	private Vocab vocab;

	private WordFilter wf;

	public KeyphraseExtractorOld(DocumentCollection dc, WordFilter wf, Counter<String> phrsWeights) {
		this.dc = dc;
		this.wf = wf;
		this.vocab = dc.getVocab();

		phrsIdxer = Generics.newIndexer(phrsWeights.size());

		{
			Trie<Integer> trie = new Trie<Integer>();

			for (String phrs : phrsWeights.keySet()) {
				List<String> words = StrUtils.split(phrs);
				List<Integer> ws = vocab.indexesOfKnown(words);

				if (ws.size() > 0 && words.size() == ws.size()) {
					Node<Integer> node = trie.insert(ws);
					node.setFlag(true);
					phrsIdxer.add(phrs);
				}
			}
			trie.trimToSize();
			pm = new PhraseMapper<>(trie);
		}

		topicBiases = new DenseVector(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double bias = phrsWeights.getCount(phrs);
			topicBiases.add(i, bias);
		}
	}

	private void buildSimMatrix(SparseVector phrsCnts) {
		T = new DenseMatrix(phrsCnts.size());

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

					// cocnt *= pos_decay;

					T.add(loc1, loc2, cocnt);
					T.add(loc2, loc1, cocnt);
				}
			}
		}
	}

	private SparseVector searchCandidatePhrases(IntegerArrayMatrix d) {
		IntegerArray d2 = DocumentCollection.toSingleSentence(d);
		List<IntPair> ps = pm.map(d2.toArrayList());
		Counter<Integer> c = Generics.newCounter();
		for (int i = 0; i < ps.size(); i++) {
			IntPair p = ps.get(i);
			IntegerArray sub = d2.subArray(p.getFirst(), p.getSecond());
			String phrs = StrUtils.join(" ", vocab.getObjects(sub));
			int pid = phrsIdxer.indexOf(phrs);
			c.incrementCount(pid, 1);
		}
		return new SparseVector(c);
	}

	public SparseVector extract(IntegerArrayMatrix d) {
		this.d = d;
		dv = DocumentCollection.toDocVector(d);

		// System.out.println(candCnts.toStringSortedByValues(true, true, candCnts.size(), "\t"));

		SparseVector phrsCnts = searchCandidatePhrases(d);
		biases = phrsCnts.copy();
		biases.setAll(0);

		buildSimMatrix();
		weightSimMatrix();

		SparseVector cents = dv.copy();
		cents.setAll(0);

		ArrayMath.randomWalk(T.values(), cents.values(), biases.values(), 100);

		// System.out.println(VectorUtils.toCounter(biases, dc.getVocab()).toStringSortedByValues(true, true, 20, "\t"));
		System.out.println(VectorUtils.toCounter(cents, dc.getVocab()).toStringSortedByValues(true, true, 50, "\t"));

		return cents;
	}

	public Counter<String> extract(String d) {

		return null;
	}

	private void weightSimMatrix() {
		for (int i = 0; i < T.size(); i++) {
			int w1 = dv.indexAt(i);
			double cnt_w1_in_d = dv.valueAt(i);
			double pr_w1_in_c = dc.getVocab().getProb(w1);
			double pr_w1_in_bg = pr_w1_in_c;
			double weight1 = TermWeighting.tfidf(cnt_w1_in_d, dc.size(), dc.getVocab().getDocFreq(w1));
			// double weight1 = TermWeighting.twoStageSmoothing(cnt_w1_in_d, len_d, pr_w1_in_c, prior_dir, pr_w1_in_bg, mixture_jm);

			for (int j = i + 1; j < T.size(); j++) {
				int w2 = dv.indexAt(j);
				double cnt_w2_in_d = dv.valueAt(j);
				double pr_w2_in_c = dc.getVocab().getProb(w2);
				double pr_w2_in_bg = pr_w2_in_c;
				double weight2 = TermWeighting.tfidf(cnt_w2_in_d, dc.size(), dc.getVocab().getDocFreq(w2));
				// double weight2 = TermWeighting.twoStageSmoothing(cnt_w2_in_d, len_d, pr_w2_in_c, prior_dir, pr_w2_in_bg, mixture_jm);
				double sim = T.value(i, j);
				double new_sim = sim * weight1 * weight2;
				T.set(i, j, new_sim);
				T.set(j, i, new_sim);
			}
		}
		T.normalizeColumns();
	}

}
