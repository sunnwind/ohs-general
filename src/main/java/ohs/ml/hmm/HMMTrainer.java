package ohs.ml.hmm;

import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.CommonMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MDocumentCollection;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.nlp.pos.NLPPath;
import ohs.nlp.pos.SejongReader;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Conditions;
import ohs.utils.Generics;
import ohs.utils.UnicodeUtils;

public class HMMTrainer {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		test01();
		// test02();

		System.out.println("process ends.");
	}

	public static void test01() {
		int[] states = { 0, 1 };
		int N = states.length;

		DenseMatrix tr_prs = new DenseMatrix(N);
		tr_prs.set(0, 0, 0.7);
		tr_prs.set(0, 1, 0.3);
		tr_prs.set(1, 0, 0.4);
		tr_prs.set(1, 1, 0.6);

		int V = 3;

		DenseMatrix ems_prs = new DenseMatrix(N, V);

		/*
		 * 산책:0, 쇼핑:1, 청소:2
		 */

		ems_prs.set(0, 0, 0.1);
		ems_prs.set(0, 1, 0.4);
		ems_prs.set(0, 2, 0.5);
		ems_prs.set(1, 0, 0.6);
		ems_prs.set(1, 1, 0.3);
		ems_prs.set(1, 2, 0.1);

		DenseVector start_prs = new DenseVector(new double[] { 0.6, 0.4 });

		HMM M = new HMM(start_prs, tr_prs, ems_prs);

		IntegerArray obs = new IntegerArray(new int[] { 0, 0, 2, 1 });

		// M.forward(obs);

		M.viterbi(obs);

		// M.backward(obs);

		HMMTrainer t = new HMMTrainer();

		t.train(M, obs, 1);

		M.viterbi(obs);
	}

	public static void test02() throws Exception {

		MDocumentCollection coll = new MDocumentCollection();

		SejongReader r = new SejongReader(NLPPath.POS_DATA_FILE);
		while (r.hasNext()) {
			if (coll.size() == 1000) {
				break;
			}
			MDocument doc = r.next();
			coll.add(doc);

		}
		r.close();

		for (MDocument doc : coll) {
			for (MSentence ts : doc) {
				for (MToken t : ts) {
					String text = t.getString(0);
					String text2 = UnicodeUtils.decomposeToJamoStr(text);

					System.out.println(text + "\t" + text2 + "\t" + String.valueOf(text2.getBytes()));
					// char[][] phomenes = UnicodeUtils.decomposeKoreanWordToPhonemes(word);
				}
			}
		}

		Set<String> posSet = Generics.newHashSet();

		for (String line : FileUtils.readLinesFromText(NLPPath.POS_TAG_SET_FILE)) {
			String[] parts = line.split("\t");
			posSet.add(parts[0]);
		}

		Indexer<String> wordIndexer = Generics.newIndexer();
		Indexer<String> posIndexer = Generics.newIndexer();

		wordIndexer.add("UNK");

		List<MSentence> sents = coll.getSentences();

		IntegerMatrix wss = new IntegerMatrix(sents.size());
		IntegerMatrix poss = new IntegerMatrix(sents.size());

		for (int i = 0; i < sents.size(); i++) {
			MSentence sent = sents.get(i);

			int[] ws = ArrayUtils.range(sent.size());
			int[] pos = ArrayUtils.range(sent.size());

			for (int j = 0; j < sent.size(); j++) {
				MToken t = sent.get(j);
				ws[j] = wordIndexer.getIndex(t.getString(0));
				pos[j] = posIndexer.getIndex(t.getString(1));
			}

			wss.add(new IntegerArray(ws));
			poss.add(new IntegerArray(pos));
		}

		HMM hmm = new HMM(posIndexer, wordIndexer);

		HMMTrainer trainer = new HMMTrainer();
		// trainer.trainUnsupervised(M, obss, 1);

		trainer.trainSupervised(hmm, wss, poss);

		hmm.write(NLPPath.POS_HMM_MODEL_FILE);

		// for (int i = 0; i < col.size(); i++) {
		// MDocument doc = col.get(i);
		//
		// for (int j = 0; j < doc.size(); j++) {
		// for (int k = 0; k < doc.size(); k++) {
		// MTSentence sent = doc.getSentence(k);
		// }
		// }
		// }

		for (int i = 0; i < wss.size(); i++) {
			IntegerArray obs = wss.get(i);
			IntegerArray sts = poss.get(i);
			IntegerArray path = hmm.viterbi(obs);

			List<String> words = Generics.newArrayList();
			List<String> anss = Generics.newArrayList();
			List<String> preds = Generics.newArrayList();

			for (int j = 0; j < obs.size(); j++) {
				String word = wordIndexer.getObject(obs.get(j));
				String pred = posIndexer.getObject(path.get(j));
				String ans = posIndexer.getObject(sts.get(j));

				words.add(word);
				preds.add(pred);
				anss.add(ans);
			}

			System.out.println(words);
			System.out.println(anss);
			System.out.println(preds);
			System.out.println();
		}
	}

	private HMM M;

	private int N;

	private int V;

	private DenseVector phi_tmp;

	private DenseMatrix a_tmp;

	private DenseMatrix b_tmp;

	public HMMTrainer() {

	}

	public DenseMatrix backward(IntegerArray x) {
		int T = x.size();
		DenseMatrix a = M.getA();
		DenseMatrix b = M.getB();
		DenseMatrix beta = new DenseMatrix(N, T);

		for (int i = 0; i < N; i++) {
			beta.set(i, T - 1, 1);
		}

		double sum = 0;
		for (int t = T - 2; t >= 0; t--) {
			for (int j = 0; j < N; j++) {
				sum = 0;
				for (int i = 0; i < N; i++) {
					sum += beta.value(i, t + 1) * a.value(j, i) * b.value(i, x.get(t + 1));
				}
				beta.set(j, t, sum);
			}
		}
		return beta;
	}

	/**
	 * 
	 * 
	 * @param x
	 * @return
	 */
	public DenseMatrix forward(IntegerArray x) {
		int T = x.size();
		DenseVector phi = M.getPhi();
		DenseMatrix a = M.getA();
		DenseMatrix b = M.getB();

		DenseMatrix alpha = new DenseMatrix(N, T);

		for (int i = 0; i < N; i++) {
			alpha.set(i, 0, phi.value(i) * b.value(i, x.get(0)));
		}
		double sum = 0;
		for (int t = 1; t < T; t++) {
			for (int j = 0; j < N; j++) {
				sum = 0;
				for (int i = 0; i < N; i++) {
					sum += alpha.value(i, t - 1) * a.value(i, j);
				}
				sum *= b.value(j, x.get(t));
				alpha.set(j, t, sum);
			}
		}
		return alpha;
	}

	public double gamma(int t, int i, DenseMatrix alpha, DenseMatrix beta) {
		double ret = (alpha.value(i, t) * beta.value(i, t));
		double norm = ArrayMath.dotProductColumns(alpha.values(), t, beta.values(), t);
		ret = CommonMath.divide(ret, norm);
		return ret;
	}

	public double likelihood(double[][] alpha) {
		return ArrayMath.sumColumn(alpha, alpha[0].length - 1);
	}

	public void train(HMM M, IntegerArray x, int iter) {
		this.M = M;
		this.N = M.getN();
		this.V = M.getV();

		phi_tmp = M.getPhi().copy(true);
		a_tmp = M.getA().copy(true);
		b_tmp = M.getB().copy(true);

		for (int i = 0; i < iter; i++) {
			train(x);
		}
	}

	public void train(IntegerArray x) {

		DenseMatrix alpha = forward(x);
		DenseMatrix beta = backward(x);

		// ArrayUtils.print("alpha", alpha);
		// ArrayUtils.print("beta", alpha);

		for (int i = 0; i < N; i++) {
			phi_tmp.set(i, gamma(0, i, alpha, beta));
		}

		int T = x.size();

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				double value = 0;
				double norm = 0;
				for (int t = 0; t < T; t++) {
					value += xi(t, i, j, x, alpha, beta);
					norm += gamma(t, i, alpha, beta);
				}
				a_tmp.set(i, j, CommonMath.divide(value, norm));
			}
		}

		for (int i = 0; i < N; i++) {
			for (int k = 0; k < V; k++) {
				double value = 0;
				double norm = 0;
				for (int t = 0; t < T; t++) {
					double g = gamma(t, i, alpha, beta);
					value += g * Conditions.value(k == x.get(t), 1, 0);
					norm += g;
				}
				b_tmp.set(i, k, CommonMath.divide(value, norm));
			}
		}

		VectorUtils.copy(phi_tmp, M.getPhi());
		VectorUtils.copy(a_tmp, M.getA());
		VectorUtils.copy(b_tmp, M.getB());

		M.print();
	}

	public void trainSupervised(HMM M, IntegerMatrix X, IntegerMatrix Y) {
		this.M = M;

		DenseVector phi = M.getPhi();
		DenseMatrix a = M.getA();
		DenseMatrix b = M.getB();

		DenseVector posPrs = new DenseVector(M.getN());
		DenseVector wordPrs = new DenseVector(M.getV());

		CounterMap<Integer, Integer> bigramCnts = Generics.newCounterMap();

		for (int i = 0; i < X.size(); i++) {
			IntegerArray x = X.get(i);
			IntegerArray y = Y.get(i);

			for (int j = 0; j < x.size(); j++) {
				int w1 = x.get(j);
				int pos1 = y.get(j);

				posPrs.add(pos1, 1);
				wordPrs.add(w1, 1);

				if (i == 0) {
					phi.add(w1, 1);
				}

				b.add(pos1, w1, 1);

				if (j + 1 < x.size()) {
					int w2 = x.get(j + 1);
					int pos2 = y.get(j + 1);
					a.add(pos1, pos2, 1);
					bigramCnts.incrementCount(w1, w2, 1);
				}
			}
		}

		phi.normalize();
		a.normalizeRows();
		posPrs.normalize();

		VectorMath.addAfterMultiply(a, 0.5, posPrs, 0.5, a);

		b.normalizeRows();
		wordPrs.normalize();

		VectorMath.addAfterMultiply(b, 0.5, wordPrs, 0.5, b);
	}

	public void trainUnsupervised(HMM M, IntegerMatrix X, int iter) {
		this.M = M;

		this.N = M.getN();
		this.V = M.getV();

		phi_tmp = new DenseVector(N);
		a_tmp = new DenseMatrix(N, N);
		b_tmp = new DenseMatrix(N, V);

		for (int i = 0; i < iter; i++) {
			for (int j = 0; j < X.size(); j++) {
				train(X.get(j));
			}
		}
	}

	public double xi(int t, int i, int j, IntegerArray x, DenseMatrix alpha, DenseMatrix beta) {
		double ret = 0;
		int T = x.size();
		DenseMatrix a = M.getA();
		DenseMatrix b = M.getB();
		if (t == T - 1) {
			ret = alpha.value(i, t) * a.value(i, j);
		} else {
			ret = alpha.value(i, t) * a.value(i, j) * b.value(j, x.get(t + 1)) * beta.value(j, t + 1);
		}
		double norm = ArrayMath.dotProductColumns(alpha.values(), t, beta.values(), t);
		ret = CommonMath.divide(ret, norm);
		return ret;
	}

}
