package ohs.eden.keyphrase.kmine;

import java.io.File;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import de.bwaldvogel.liblinear.Linear;
import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListList;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;


/**
 * Rank candidate phrases using TextRank algorithm
 * 
 * @author ohs
 *
 */
public class KPhraseRanker {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsCnts = FileUtils.readStringCounterFromText(KPPath.KP_DIR + "ext/phrs.txt");

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		Vocab featIdxer = DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser");

		KPhraseRanker pr = new KPhraseRanker(featIdxer,
				KCandidatePhraseSearcher.newCandidatePhraseSearcher(phrsCnts.keySet()));

		KPhraseNumberPredictor pnp = new KPhraseNumberPredictor(featIdxer,
				Linear.loadModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt")));
		int cnt = 0;

		for (String line : ins) {
			List<String> ps = StrUtils.split("\t", line);

			List<String> ansPhrss = Generics.newArrayList();

			MDocument md1 = MDocument.newDocument(ps.get(1));
			MDocument md2 = MDocument.newDocument(ps.get(2));

			Counter<MSentence> phrss = pr.rank(md2);

			int pred_phrs_size = pnp.predict(md2);

			phrss.keepTopNKeys(pred_phrs_size);

			System.out.println("<Input>");
			// System.out.println(sb1.toString());
			System.out.println("<Output>");

			StringBuffer sb2 = new StringBuffer();

			for (MSentence mt : phrss.getSortedKeys()) {
				StringBuffer sb = new StringBuffer();
				double score = phrss.getCount(mt);
				for (Token t : mt.getTokens()) {
					sb.append(t.get(0));
				}

			}

			System.out.println(phrss.toStringSortedByValues(true, true, phrss.size(), "\t"));
			System.out.println();

			// ke.extractTFIDF(content);

			if (++cnt == 30) {
				break;
			}
		}

		int dseq = 1000;

		// ke.extract(StrUtils.split(text));

		// ke.extractLM(StrUtils.split(text));

		System.out.println("process ends.");
	}

	private KCandidatePhraseSearcher cps;

	private Vocab featIdxer;

	private WordFilter wf;

	private int window_size = 5;

	public KPhraseRanker(Vocab featIdxer, KCandidatePhraseSearcher cps) {
		this.featIdxer = featIdxer;
		this.cps = cps;
	}

	private Counter<MSentence> getPhraseCounts(MDocument doc, List<List<IntPair>> psData) {
		Counter<MSentence> ret = Generics.newCounter();
		for (int i = 0; i < doc.size(); i++) {
			List<Token> ts = doc.get(i).getTokens();
			List<IntPair> ps = psData.get(i);

			for (IntPair p : ps) {
				MSentence phrs = new MSentence();

				for (int j = p.getFirst(); j < p.getSecond(); j++) {
					MultiToken mt = new MultiToken();
					mt.add(ts.get(j));
					phrs.add(mt);
				}

				ret.incrementCount(phrs, 1);
			}
		}

		return ret;
	}

	private CounterMap<MSentence, Token> getPhraseToWords(MDocument d, List<List<IntPair>> psData) {
		CounterMap<MSentence, Token> ret = Generics.newCounterMap();

		for (int i = 0; i < d.size(); i++) {
			MSentence s = d.get(i);
			List<IntPair> ps = psData.get(i);

			if (ps.size() == 0) {
				continue;
			}

			List<Token> ts = s.getTokens();

			for (IntPair p : ps) {
				MSentence phrs = new MSentence();

				for (int j = p.getFirst(); j < p.getSecond(); j++) {
					MultiToken mt = new MultiToken();
					mt.add(ts.get(j));
					phrs.add(mt);
				}

				int lower_boud = Math.max(0, p.getFirst() - window_size);
				int upper_bound = Math.min(p.getSecond() + window_size, ts.size());

				for (int j = lower_boud; j < p.getFirst(); j++) {
					double dist = p.getFirst() - j;
					double score = 1d / dist;
					Token t = ts.get(j);
					ret.incrementCount(phrs, t, score);
				}

				for (int j = p.getSecond(); j < upper_bound; j++) {
					double dist = j - p.getSecond() + 1;
					double score = 1d / dist;
					Token t = ts.get(j);
					ret.incrementCount(phrs, t, score);
				}
			}

		}

		return ret;
	}

	private DenseVector getPhraseWeights(Indexer<MSentence> phrsIdxer, Indexer<Token> wordIdxer,
			DenseVector wordWeights) {
		DenseVector ret = new DenseVector(phrsIdxer.size());
		for (int p = 0; p < phrsIdxer.size(); p++) {
			MSentence phrs = phrsIdxer.getObject(p);
			double weight = 0;
			for (Token t : phrs.getTokens()) {
				int w = wordIdxer.indexOf(t);
				weight += wordWeights.value(w);
			}
			ret.add(p, weight);
		}
		return ret;
	}

	public List<String> getWords(String content) {
		MDocument doc = MDocument.newDocument(content);
		List<String> ret = Generics.newArrayList();
		for (MSentence sent : doc) {
			for (MultiToken mt : sent) {
				for (Token t : mt) {
					ret.add(String.format("%s_/_%s", t.get(TokenAttr.WORD), t.get(TokenAttr.POS)));
				}
			}
		}
		return ret;
	}

	private CounterMap<Token, Token> getWordToWords(MDocument doc) {
		CounterMap<Token, Token> ret = Generics.newCounterMap();
		for (int i = 0; i < doc.size(); i++) {
			MSentence s = doc.get(i);
			List<Token> ts = s.getTokens();

			for (int j = 0; j < ts.size(); j++) {
				Token t1 = ts.get(j);

				double score1 = 1d / (j + 1);
				// phrsBiases.incrementCount(t1, score1);

				int m = Math.min(i + window_size, ts.size());

				for (int k = j + 1; k < m; k++) {
					Token t2 = ts.get(k);
					double dist = k - j;
					double score2 = 1d / dist;
					ret.incrementCount(t1, t2, score2);
				}
			}
		}
		return ret;
	}

	private DenseVector getWordWeights(Indexer<Token> wordIdxer, Counter<Token> wordCnts) {
		DenseVector ret = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			Token t = wordIdxer.getObject(w);
			String word = t.get(0);
			double doc_freq = featIdxer.getDocFreq(word);
			double cnt = wordCnts.getCount(t);
			if (doc_freq == 0) {
				// doc_freq = 1;
				doc_freq = featIdxer.getDocCnt();
			}
			double tfidf = TermWeighting.tfidf(cnt, featIdxer.getDocCnt() + 1, doc_freq);
			ret.add(w, tfidf);
		}
		return ret;
	}

	public Counter<MSentence> rank(MDocument doc) {
		List<List<IntPair>> pss = cps.search(doc);
		CounterMap<MSentence, Token> phrsSims = getPhraseToWords(doc, pss);
		CounterMap<Token, Token> wordSims = getWordToWords(doc);
		Counter<MSentence> phrsCnts = getPhraseCounts(doc, pss);
		Counter<Token> wordCnts = Generics.newCounter();

		for (List<Token> ts : doc.getTokens()) {
			for (Token t : ts) {
				wordCnts.incrementCount(t, 1);
			}
		}

		Indexer<MSentence> phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
		Indexer<Token> wordIdxer = Generics.newIndexer(wordCnts.keySet());

		DenseVector wordWeights = getWordWeights(wordIdxer, wordCnts);
		DenseVector phrsWeights = getPhraseWeights(phrsIdxer, wordIdxer, wordWeights);
		DenseVector pBiases = new DenseVector(phrsIdxer.size());

		// System.out.println(cm1.toString());

		DenseMatrix T1 = new DenseMatrix(phrsIdxer.size(), wordIdxer.size());

		for (MSentence phrs : phrsSims.keySet()) {
			int p = phrsIdxer.indexOf(phrs);
			double p_weight = phrsWeights.value(p);
			Counter<Token> c = phrsSims.getCounter(phrs);

			for (Entry<Token, Double> e : c.entrySet()) {
				Token word = e.getKey();
				double sim = e.getValue();
				int w = wordIdxer.indexOf(word);
				double w_weight = wordWeights.value(w);
				double new_sim = sim * p_weight * w_weight;
				T1.add(p, w, new_sim);
			}
		}
		// T1.normalizeColumns();

		// VectorMath.unitVector(T1, T1);

		DenseMatrix T2 = new DenseMatrix(wordIdxer.size());

		for (Token t1 : wordSims.keySet()) {
			int w1 = wordIdxer.indexOf(t1);
			double weight1 = wordWeights.value(w1);
			Counter<Token> c = wordSims.getCounter(t1);

			for (Entry<Token, Double> e : c.entrySet()) {
				Token t2 = e.getKey();
				double sim = e.getValue();
				int w2 = wordIdxer.indexOf(t2);
				double weight2 = wordWeights.value(w2);
				double new_sim = sim * weight1 * weight2;

				T2.add(w1, w2, new_sim);
				T2.add(w2, w1, new_sim);
			}
		}

		// T2.normalizeRows();

		DenseMatrix T3 = VectorMath.product(T1, T2);

		DenseMatrix T4 = new DenseMatrix(T3.rowSize());

		VectorMath.productRows(T3, T3, T4, false);

		T4.normalizeColumns();

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = pBiases.copy();
		DenseVector biases2 = cents.copy();

		biases.normalize();

//		for (List<IntPair> ps : pss) {
//			for (IntPair p : ps) {
//				String phrs = StrUtils.join(LONG_UNDER_BAR, d.subList(p.getFirst(), p.getSecond()));
//				int pos = p.getFirst();
//				int pid = phrsIdxer.indexOf(phrs);
//				biases2.add(pid, 1d / (pos + 1));
//			}
		// }
	

		// ArrayMath.addAfterMultiply(biases.values(), 0.5, biases2.values(), 0.5,
		// biases2.values());
		// ArrayMath.multiply(biases.values(), biases2.values(), biases2.values());

		biases2.normalizeAfterSummation();

		ArrayMath.randomWalk(T4.values(), cents.values(), null, 500, 10);

		Counter<MSentence> ret = Generics.newCounter();

		for (int w = 0; w < phrsIdxer.size(); w++) {
			MSentence phrs = phrsIdxer.getObject(w);
			ret.incrementCount(phrs, cents.value(w));
		}

		// System.out.println(VectorUtils.toCounter(cents,
		// phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		return ret;
	}

}
