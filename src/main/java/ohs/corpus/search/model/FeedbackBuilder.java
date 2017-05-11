package ohs.corpus.search.model;

import java.util.List;
import java.util.Map;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.app.WordSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.WordFilter;
import ohs.corpus.type.DocumentCollection;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class FeedbackBuilder {

	private DocumentCollection dc;

	private InvertedIndex ii;

	private Vocab vocab;

	private int fb_doc_size = 10;

	private int fb_word_size = 30;

	private double prior_dir = 2000;

	private double mixture_jm = 0;

	private double mixture_fb = 0.5;

	private WordFilter filter;

	private Map<Integer, Double> cache;

	private DocumentPriorEstimator dpe;

	private boolean use_doc_prior = false;

	private int len_psg_fix = 1000;

	private int window_size_psg = 500;

	private boolean print_log = false;

	private DenseVector lm_qbg;

	public FeedbackBuilder(Vocab vocab, DocumentCollection dc, InvertedIndex ii, WordFilter filter) {
		super();
		this.vocab = vocab;
		this.dc = dc;
		this.ii = ii;
		this.filter = filter;

		cache = Generics.newHashMap();

		dpe = new DocumentPriorEstimator(vocab, dc, filter);
		dpe.setDirichletPrior(prior_dir);
		dpe.setMixtureJM(mixture_jm);
	}

	public SparseVector buildCBEEM(List<DocumentSearcher> dss, List<SparseVector> docScoreData) throws Exception {
		int col_size = dss.size();
		DoubleArray colPriors = new DoubleArray(col_size);
		List<Counter<String>> rms = Generics.newArrayList(col_size);

		double len_e = 0;
		for (DocumentSearcher ds : dss) {
			len_e += ds.getVocab().sizeOfTokens();
		}

		for (int i = 0; i < dss.size(); i++) {
			DocumentSearcher ds = dss.get(i);
			Vocab vocab = ds.getVocab();
			WordFilter filter = ds.getWordFilter();

			SparseVector docScores = docScoreData.get(i).copy();
			docScores.keepTopN(fb_doc_size);

			SparseMatrix docVecs = ds.getDocumentCollection().getDocVectors(docScores.indexes());
			SparseVector cnts_w_in_fb = getWordCounts(docVecs);

			Counter<Integer> lm_fb = Generics.newCounter();

			SparseVector docPriors = null;

			if (use_doc_prior) {
				docPriors = dpe.estimateUsingLM(docScores.indexes());
			}

			double pr_w_in_d_jm = 0;
			double pr_w_in_fb = 0;
			double pr_w_in_c = 0;
			double pr_w_in_e = 0;
			double doc_weight = 0;
			double cnt_w_in_d = 0;
			double cnt_w_in_e = 0;
			double len_d = 0;
			double doc_prior = 1;

			int w = 0;
			int docseq = 0;
			String word = null;

			for (int j = 0; j < cnts_w_in_fb.size(); j++) {
				w = cnts_w_in_fb.indexAt(j);

				if (filter.filter(w)) {
					continue;
				}

				word = vocab.getObject(w);
				pr_w_in_c = vocab.getProb(w);

				cnt_w_in_e = 0;

				for (DocumentSearcher o : dss) {
					cnt_w_in_e += o.getVocab().getCount(word);
				}

				pr_w_in_e = cnt_w_in_e / len_e;

				for (int k = 0; k < docVecs.size(); k++) {
					docseq = docScores.indexAt(k);
					doc_weight = docScores.valueAt(k);

					SparseVector dv = docVecs.get(k);
					cnt_w_in_d = dv.value(w);
					len_d = dv.sum();

					pr_w_in_d_jm = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_e, mixture_jm);

					doc_prior = 1;

					if (docPriors != null) {
						doc_prior = docPriors.valueAt(k);
					}

					pr_w_in_fb = doc_weight * pr_w_in_d_jm * doc_prior;

					if (pr_w_in_fb > 0) {
						lm_fb.incrementCount(w, pr_w_in_fb);
					}
				}
			}

			lm_fb.keepTopNKeys(fb_word_size);
			lm_fb.normalize();

			rms.add(VectorUtils.toCounter(lm_fb, vocab));

			double col_prior = ArrayMath.sum(docScores.values());
			colPriors.add(col_prior);
		}

		ArrayMath.normalize(colPriors.values());

		Counter<String> rmm = Generics.newCounter();

		StringBuffer sb = new StringBuffer();

		if (print_log) {
			sb.append("<EEM>");
			for (int i = 0; i < rms.size(); i++) {
				double col_prior = colPriors.get(i);
				Counter<String> rm = rms.get(i);
				sb.append(String.format("\n%d, %f, %s", i, col_prior, rm.toString()));
				for (String word : rm.keySet()) {
					rmm.incrementCount(word, col_prior * rm.getCount(word));
				}
			}
		}

		rmm.keepTopNKeys(fb_word_size);
		rmm.normalize();

		if (print_log) {
			sb.append(String.format("\n%d, %f, %s", rms.size(), 1f, rmm.toString()));
			System.out.println(sb.toString() + "\n");
		}

		return VectorUtils.toSparseVector(rmm, dss.get(0).getVocab());
	}

	/**
	 * 
	 * Diaz, F., & Metzler, D. (2006). Improving the estimation of relevance models using large external corpora. In Proceedings of the 29th
	 * annual international ACM SIGIR conference on Research and development in information retrieval - SIGIR ’06 (p. 154). New York, New
	 * York, USA: ACM Press. http://doi.org/10.1145/1148170.1148200
	 * 
	 * Weerkamp, W., Balog, K., & Rijke, M. de. (2009). A generative blog post retrieval model that uses query expansion based on external
	 * collections. In Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference
	 * on Natural Language Processing of the AFNLP (Vol. 2, pp. 1057–1065). Retrieved from http://dl.acm.org/citation.cfm?id=1690294
	 * 
	 * 
	 * @param dss
	 * @param scoreData
	 * @return
	 * @throws Exception
	 */
	public SparseVector buildEEM(List<DocumentSearcher> dss, List<SparseVector> scoreData) throws Exception {
		int sizes_col = dss.size();
		double[] priors_col = new double[sizes_col];
		double[] lens_fb = new double[sizes_col];

		List<Counter<String>> rms = Generics.newArrayList(sizes_col);
		Vocab vocab = dss.get(0).getVocab();

		for (int i = 0; i < dss.size(); i++) {
			DocumentSearcher ds = dss.get(i);
			SparseVector scores = scoreData.get(i);
			SparseVector rm = buildRM1(scores);
			rms.add(VectorUtils.toCounter(rm, vocab));

			int size_doc_fb = Integer.min(scores.size(), this.fb_doc_size);
			priors_col[i] = scores.subVector(size_doc_fb).sum();

			double len = 0;
			for (int j = 0; j < size_doc_fb; j++) {
				len += ds.getDocumentCollection().getDocLength(scores.indexAt(j));
			}
			lens_fb[i] = len;
		}

		// ArrayMath.multiply(col_priors, lens_fb, col_priors);
		// ArrayMath.normalize(col_priors);

		Counter<String> c = Generics.newCounter();

		StringBuffer sb = new StringBuffer();

		if (print_log) {
			sb.append("<EEM>");
		}

		for (int i = 0; i < rms.size(); i++) {
			double col_prior = priors_col[i];
			Counter<String> rm = rms.get(i);
			if (print_log) {
				sb.append(String.format("\n%d, %f, %s", i, col_prior, rm.toString()));
			}
			for (String word : rm.keySet()) {
				c.incrementCount(word, col_prior * rm.getCount(word));
			}
		}

		c.keepTopNKeys(fb_word_size);
		c.normalize();

		if (print_log) {
			sb.append(String.format("\n%d, %f, %s", rms.size(), 1f, c.toString()));
			System.out.println(sb.toString() + "\n");
		}

		return VectorUtils.toSparseVector(c, vocab);
	}

	public SparseVector buildEmbeddingRM1(SparseVector lm_q, SparseVector scores, WordSearcher ws) throws Exception {
		scores = scores.subVector(Math.min(fb_doc_size, scores.size()));

		SparseMatrix dvs = dc.getDocVectors(scores.indexes());
		SparseVector lm_fb = getWordCounts(dvs);

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(scores.indexes());
			// docPriors = dpe.estimateUsingIDF(docScores.indexes());
		}

		DenseMatrix m1 = new DenseMatrix(lm_q.size(), lm_fb.size());

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < lm_q.size(); i++) {
			int w1 = lm_q.indexAt(i);

			double cnt1 = lm_q.valueAt(i);
			double idf1 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w1));
			// double pr_w1_in_c = vocab.getProb(w1);
			// double pr_w1 = TermWeighting.twoStageSmoothing(cnt1, lm_fb.sum(),
			// pr_w1_in_c, prior_dir, pr_w1_in_c, mixture_jm);

			for (int j = 0; j < lm_fb.size(); j++) {
				int w2 = lm_fb.indexAt(j);

				if (filter.filter(w2)) {
					continue;
				}

				double cnt2 = lm_fb.valueAt(j);
				double pr_w2_in_c = vocab.getProb(w2);
				double pr_w2 = TermWeighting.twoStageSmoothing(cnt2, lm_fb.sum(), pr_w2_in_c, prior_dir, pr_w2_in_c, mixture_jm);
				double idf2 = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w2));
				double cosine = ws.getCosine(w1, w2);

				cosine = Math.max(0, cosine);
				// cosine = CommonMath.sigmoid(cosine);
				cosine *= idf1 * idf2;

				m1.add(i, j, cosine);

				cm.setCount(w1, w2, cosine);
			}
		}

		m1.normalizeRows();

		DenseMatrix m2 = VectorMath.transpose(m1);

		DenseMatrix m3 = VectorMath.product(m2, m1);

		SparseVector cents = lm_fb.copy();

		ArrayMath.randomWalk(m3.values(), cents.values(), 30, 0.000001, 1);

		for (int i = 0; i < lm_fb.size(); i++) {
			int w = lm_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			double pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < dvs.size(); j++) {
				int docseq = scores.indexAt(j);
				double weight_d = scores.valueAt(j);
				SparseVector dv = dvs.get(j);
				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				double prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				double pr_w_in_fb = weight_d * pr_w_in_d * prior_d;

				if (pr_w_in_fb > 0) {
					lm_fb.addAt(i, pr_w_in_fb);
				}
			}
		}

		ArrayMath.multiply(lm_fb.values(), cents.values(), lm_fb.values());

		lm_fb.sortValues();
		lm_fb = lm_fb.subVector(fb_word_size);
		lm_fb.normalize();
		return lm_fb;
	}

	public SparseVector buildEmbeddingRM2(SparseVector lm_q, SparseVector scores, WordSearcher ws) throws Exception {
		scores = scores.subVector(Math.min(fb_doc_size, scores.size()));

		SparseMatrix dvs = dc.getDocVectors(scores.indexes());
		SparseVector lm_fb = getWordCounts(dvs);

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(scores.indexes());
			// docPriors = dpe.estimateUsingIDF(docScores.indexes());
		}

		DenseMatrix m1 = new DenseMatrix(scores.size(), ws.getEmbeddingMatrix().colSize());

		DenseVector eq = new DenseVector(ws.getEmbeddingMatrix().colSize());

		for (int j = 0; j < lm_q.size(); j++) {
			int w = lm_q.indexAt(j);
			double cnt = lm_q.valueAt(j);
			VectorMath.add(ws.getVector(w), eq);
		}

		SparseVector scores2 = scores.copy();

		for (int i = 0; i < scores.size(); i++) {
			int docseq = scores.indexAt(i);
			SparseVector d = dvs.rowAt(i);

			DenseVector ed = m1.row(i);

			for (int j = 0; j < d.size(); j++) {
				int w = d.indexAt(j);
				double cnt = d.valueAt(j);
				double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
				// VectorMath.add(ws.getVector(w), cnt, ed);
				VectorMath.addAfterMultiply(ws.getVector(w), cnt, ed);
			}

			double cosine = VectorMath.cosine(eq, ed);

			cosine = Math.exp(cosine);

			scores2.setAt(i, cosine);
		}

		scores2.sortValues();

		scores = scores2;

		lm_fb.setAll(0);

		for (int i = 0; i < lm_fb.size(); i++) {
			int w = lm_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			double pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < dvs.size(); j++) {
				int docseq = scores.indexAt(j);
				double weight_d = scores.valueAt(j);
				SparseVector dv = dvs.get(j);
				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				double prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				double pr_w_in_fb = weight_d * pr_w_in_d * prior_d;

				if (pr_w_in_fb > 0) {
					lm_fb.addAt(i, pr_w_in_fb);
				}
			}
		}

		lm_fb.sortValues();
		lm_fb = lm_fb.subVector(fb_word_size);
		lm_fb.normalize();
		return lm_fb;
	}

	public SparseVector buildNegRM1(SparseVector docScores, int start) throws Exception {
		SparseVector rm1 = buildRM1(docScores, 0, null);
		SparseVector rm2 = buildRM1(docScores, 100, null);
		SparseVector rm3 = VectorMath.addAfterMultiply(rm1, 1, rm2, -1);

		for (int k = 0; k < rm3.size(); k++) {
			double pr = rm3.valueAt(k);
			rm3.setAt(k, Math.exp(pr * 10));
		}

		rm3.sortValues();
		rm3 = rm3.subVector(fb_word_size);
		rm3.normalize();
		return rm3;
	}

	public SparseVector buildPassageRM1(SparseVector lm_q, SparseVector docScores) throws Exception {
		docScores = docScores.copy();

		docScores.keepTopN(fb_doc_size);

		SparseMatrix docVecs = dc.getDocVectors(docScores.indexes());

		SparseVector cnts_w_in_fb = getWordCounts(docVecs);

		Counter<Integer> lm_fb = Generics.newCounter();

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(docScores.indexes());
		}

		List<SparseMatrix> psgData = Generics.newArrayList(docScores.size());
		SparseMatrix psgScoreData = new SparseMatrix();

		double dirichlet_prior = 0;
		double mixture_jm = 0.5;

		for (int docseq : docScores.indexes()) {
			SparseMatrix psgs = getPassageVectors(docseq);
			SparseVector psgScores = new SparseVector(psgs.size());

			int loc = 0;
			for (SparseVector psg : psgs) {
				psgScores.addAt(loc, loc, LMScorer.score(lm_q, psg, vocab, dirichlet_prior, mixture_jm));
				loc++;
			}

			psgScores.normalize();

			DenseVector posWeights = new DenseVector(psgScores.size());

			for (int i = 0; i < posWeights.size(); i++) {
				posWeights.add(i, Math.exp(-i));
			}

			posWeights.normalize();

			ArrayMath.multiply(posWeights.values(), psgScores.values(), psgScores.values());
			psgScores.normalize();

			psgScores.sortValues();

			psgData.add(psgs);
			psgScoreData.add(psgScores);
		}

		double prior_d = 1;
		double weight_d = 0;
		double weight_psg = 0;
		double pr_w_in_c = 0;
		double pr_w_in_psg_jm = 0;
		double pr_w_in_fb = 0;
		double cnt_w_in_psg = 0;
		double len_psg = 0;
		int w = 0;
		int docseq = 0;
		int psgseq = 0;
		int fb_psg_size = psgScoreData.size();

		for (int i = 0; i < cnts_w_in_fb.size(); i++) {
			w = cnts_w_in_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < docScores.size(); j++) {
				docseq = docScores.indexAt(j);
				weight_d = docScores.valueAt(j);
				SparseMatrix psgs = psgData.get(j);
				SparseVector psgScores = psgScoreData.get(j);

				prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				for (int k = 0; k < psgScores.size() && k < fb_psg_size; k++) {
					psgseq = psgScores.indexAt(k);
					weight_psg = psgScores.valueAt(k);

					SparseVector psg = psgs.row(psgseq);
					cnt_w_in_psg = psg.value(w);
					len_psg = psg.sum();

					pr_w_in_psg_jm = TermWeighting.twoStageSmoothing(cnt_w_in_psg, len_psg, pr_w_in_c, dirichlet_prior, pr_w_in_c,
							mixture_jm);

					pr_w_in_fb = weight_d * weight_psg * pr_w_in_psg_jm * prior_d;
					// pr_w_in_fb = weight_psg * pr_w_in_psg_jm * prior_d;

					if (pr_w_in_fb > 0) {
						lm_fb.incrementCount(w, pr_w_in_fb);
					}
				}
			}
		}

		lm_fb.keepTopNKeys(fb_word_size);
		lm_fb.normalize();

		SparseVector ret = VectorUtils.toSparseVector(lm_fb);

		return ret;
	}

	public SparseVector buildPassageRM2(SparseVector lm_q, SparseVector docScores) throws Exception {
		docScores = docScores.copy();

		docScores.keepTopN(fb_doc_size);

		SparseMatrix docVecs = dc.getDocVectors(docScores.indexes());

		SparseVector cnts_w_in_fb = getWordCounts(docVecs);

		Counter<Integer> lm_fb = Generics.newCounter();

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(docScores.indexes());
		}

		List<SparseMatrix> psgData = Generics.newArrayList(docScores.size());
		SparseMatrix psgScoreData = new SparseMatrix();

		double dirichlet_prior = 0;
		double mixture_jm = 0.5;

		for (int docseq : docScores.indexes()) {
			SparseMatrix psgs = getPassageVectors(docseq);
			SparseVector psgScores = new SparseVector(psgs.size());

			int loc = 0;
			for (SparseVector psg : psgs) {
				psgScores.addAt(loc, loc, LMScorer.score(lm_q, psg, vocab, dirichlet_prior, mixture_jm));
				loc++;
			}

			DenseVector posWeights = new DenseVector(psgScores.size());

			for (int i = 0; i < posWeights.size(); i++) {
				posWeights.add(i, 1f / (i + 1));
			}

			// int center = posWeights.size() / 2;

			// for (int i = 0; i < center; i++) {
			// int dist = center - i;
			// double weight = 1f / (dist + 1);
			// posWeights.add(i, weight);
			// }
			//
			// for (int i = center; i < psgs.size(); i++) {
			// int dist = i - center;
			// double weight = 1f / (dist + 1);
			// posWeights.add(i, weight);
			// }

			// ArrayMath.multiply(posWeights.values(), psgScores.values(),
			// psgScores.values());

			psgScores.sortValues();

			psgData.add(psgs);
			psgScoreData.add(psgScores);
		}

		double prior_d = 1;
		double weight_d = 0;
		double weight_psg = 0;
		double pr_w_in_c = 0;
		double pr_w_in_psg_jm = 0;
		double pr_w_in_fb = 0;
		double cnt_w_in_psg = 0;
		double len_psg = 0;
		int w = 0;
		int docseq = 0;
		int psgseq = 0;
		int fb_psg_size = psgScoreData.size();

		for (int i = 0; i < cnts_w_in_fb.size(); i++) {
			w = cnts_w_in_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < docScores.size(); j++) {
				docseq = docScores.indexAt(j);
				weight_d = docScores.valueAt(j);
				SparseMatrix psgs = psgData.get(j);
				SparseVector psgScores = psgScoreData.get(j);

				prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				for (int k = 0; k < psgScores.size() && k < fb_psg_size; k++) {
					psgseq = psgScores.indexAt(k);
					weight_psg = psgScores.valueAt(k);

					SparseVector psg = psgs.row(psgseq);
					cnt_w_in_psg = psg.value(w);
					len_psg = psg.sum();

					pr_w_in_psg_jm = TermWeighting.twoStageSmoothing(cnt_w_in_psg, len_psg, pr_w_in_c, dirichlet_prior, pr_w_in_c,
							mixture_jm);

					pr_w_in_fb = weight_d * weight_psg * pr_w_in_psg_jm * prior_d;
					// pr_w_in_fb = weight_psg * pr_w_in_psg_jm * prior_d;

					if (pr_w_in_fb > 0) {
						lm_fb.incrementCount(w, pr_w_in_fb);
					}
				}
			}
		}

		lm_fb.keepTopNKeys(fb_word_size);
		lm_fb.normalize();

		return VectorUtils.toSparseVector(lm_fb);
	}

	public SparseVector buildRM1(SparseVector scores) throws Exception {
		return buildRM1(scores, 0, null);
	}

	public SparseVector buildRM1(SparseVector scores, int start, DenseVector docPriors) throws Exception {
		scores = scores.subVector(start, Math.min(fb_doc_size, scores.size() - start));

		SparseMatrix dvs = dc.getDocVectors(scores.indexes());
		SparseVector lm_fb = getWordCounts(dvs);

		lm_fb.setAll(0);

		for (int i = 0; i < lm_fb.size(); i++) {
			int w = lm_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			double pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < scores.size(); j++) {
				int dseq = scores.indexAt(j);
				double weight_d = scores.valueAt(j);
				SparseVector dv = dvs.row(dseq);
				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				double prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.value(dseq);
				}

				double pr_w_in_fb = weight_d * pr_w_in_d * prior_d;

				if (pr_w_in_fb > 0) {
					lm_fb.addAt(i, pr_w_in_fb);
				}
			}
		}

		lm_fb.sortValues();
		lm_fb = lm_fb.subVector(fb_word_size);
		lm_fb.normalize();
		return lm_fb;
	}

	public SparseVector buildSpecificLM(SparseVector docScores) throws Exception {
		docScores = docScores.copy();

		docScores.keepTopN(fb_doc_size);

		SparseMatrix docVecs = dc.getDocVectors(docScores.indexes());
		SparseVector cnts_w_in_fb = getWordCounts(docVecs);

		Counter<Integer> lm_fb = Generics.newCounter();

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(docScores.indexes());
		}

		double pr_w_in_d = 0;
		double pr_w_out_d = 0;
		double pr_w_in_fb = 0;

		int w = 0;
		int docseq1 = 0;
		int docseq2 = 0;

		for (int i = 0; i < cnts_w_in_fb.size(); i++) {
			w = cnts_w_in_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			for (int j = 0; j < docScores.size(); j++) {
				docseq1 = docScores.indexAt(j);
				SparseVector dv1 = dc.getDocVector(docseq1);
				pr_w_in_d = dv1.prob(w);

				pr_w_out_d = 0;

				for (int k = 0; k < docScores.size(); k++) {
					if (j == k) {
						continue;
					}

					docseq2 = docScores.indexAt(k);
					SparseVector dv2 = dc.getDocVector(docseq2);
					pr_w_out_d += (1 - dv2.prob(w));
				}

				pr_w_in_fb += pr_w_in_d * pr_w_out_d;
			}

			lm_fb.setCount(w, pr_w_in_fb);
		}

		// System.out.println(new IntegerArray(dvs.rowIndexes()));
		// System.out.println(VectorUtils.toCounter(wcnts, vocab));

		lm_fb.keepTopNKeys(fb_word_size);
		lm_fb.normalize();

		// System.out.println(VectorUtils.toCounter(lm_fb, vocab));

		return VectorUtils.toSparseVector(lm_fb);
	}

	public SparseVector buildTRM1(SparseVector scores, int start) throws Exception {
		scores = scores.subVector(start, Math.min(fb_doc_size, scores.size() - start));

		SparseMatrix dvs = dc.getDocVectors(scores.indexes());
		SparseVector lm_fb = getWordCounts(dvs);

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(scores.indexes());
			// docPriors = dpe.estimateUsingIDF(docScores.indexes());
		}

		TranslationModelBuilder dmb = new TranslationModelBuilder(vocab, dc, null, filter);
		SparseMatrix lms_d = dmb.buildTranslatedModels(dvs);

		SparseMatrix T = dmb.getTranslationModel(dmb.getProximities(scores.indexes()), lm_fb);

		SparseVector lm_fb_mle = lm_fb.copy();
		lm_fb_mle.normalize();

		SparseVector lm_fb_trs = dmb.translateModel(T, lm_fb_mle);

		// CounterMap<Integer, Integer> cm1 =
		// dmb.getProximities(docScores.indexes());
		// System.out.println(VectorUtils.toCounterMap(cm1, vocab, vocab));

		lm_fb.setAll(0);

		for (int i = 0; i < lm_fb.size(); i++) {
			int w = lm_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			double pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < lms_d.size(); j++) {
				int docseq = scores.indexAt(j);
				double weight_d = scores.valueAt(j);

				SparseVector lm_d = lms_d.rowAt(j);

				double pr_w_in_d = lm_d.value(w);
				double pr_w_in_d_jm = pr_w_in_d = TermWeighting.jelinekMercerSmoothing(pr_w_in_d, pr_w_in_c, mixture_jm);
				double prior_d = 1;

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				double pr_w_in_fb = weight_d * pr_w_in_d_jm * prior_d;

				if (pr_w_in_fb > 0) {
					lm_fb.addAt(i, pr_w_in_fb);
				}
			}
		}

		lm_fb.sortValues();
		lm_fb = lm_fb.subVector(fb_word_size);
		lm_fb.normalize();
		return lm_fb;
	}

	public SparseVector buildTRM2(SparseVector docScores, int start) throws Exception {
		docScores = docScores.subVector(start, Math.min(fb_doc_size, docScores.size() - start));

		SparseMatrix dvs = dc.getDocVectors(docScores.indexes());
		SparseVector lm_fb = getWordCounts(dvs);

		SparseVector docPriors = null;

		if (use_doc_prior) {
			docPriors = dpe.estimateUsingLM(docScores.indexes());
			// docPriors = dpe.estimateUsingIDF(docScores.indexes());
		}

		TranslationModelBuilder dmb = new TranslationModelBuilder(vocab, dc, null, filter);
		SparseMatrix lms_d_trs = dmb.buildTranslatedModels(dvs);

		SparseMatrix T = dmb.getTranslationModel(dmb.getProximities(docScores.indexes()), lm_fb);

		List<Integer> idxs = Generics.newArrayList();
		List<SparseVector> rows = Generics.newArrayList();

		for (int i = 0; i < dvs.size(); i++) {
			int docseq = dvs.indexAt(i);
			SparseVector lm_d = dvs.rowAt(i).copy();
			lm_d.normalize();
			lm_d = dmb.translateModel(T, lm_d);
			idxs.add(docseq);
			rows.add(lm_d);
		}

		SparseMatrix lms_d_trs_g = new SparseMatrix(idxs, rows);

		SparseVector lm_fb_mle = lm_fb.copy();
		lm_fb_mle.normalize();

		// SparseVector lm_fb_trs = dmb.translateModel(T, lm_fb_mle);

		// CounterMap<Integer, Integer> cm1 =
		// dmb.getProximities(docScores.indexes());
		// System.out.println(VectorUtils.toCounterMap(cm1, vocab, vocab));

		lm_fb.setAll(0);

		for (int i = 0; i < lm_fb.size(); i++) {
			int w = lm_fb.indexAt(i);

			if (filter.filter(w)) {
				continue;
			}

			double pr_w_in_c = vocab.getProb(w);

			for (int j = 0; j < lms_d_trs.size(); j++) {
				int docseq = docScores.indexAt(j);
				double weight_d = docScores.valueAt(j);

				SparseVector lm_d = lms_d_trs.rowAt(j);
				SparseVector lm_d_trs_g = lms_d_trs_g.rowAt(j);

				double pr_w_in_d_trs = lm_d.value(w);
				pr_w_in_d_trs = TermWeighting.jelinekMercerSmoothing(pr_w_in_d_trs, pr_w_in_c, mixture_jm);
				double prior_d = 1;

				double pr_w_in_d_trs_g = lm_d_trs_g.value(w);

				double pr_w_in_d = ArrayMath.addAfterMultiply(pr_w_in_d_trs, 0.5, pr_w_in_d_trs_g);

				if (docPriors != null) {
					prior_d = docPriors.valueAt(j);
				}

				double pr_w_in_fb = weight_d * pr_w_in_d * prior_d;

				if (pr_w_in_fb > 0) {
					lm_fb.addAt(i, pr_w_in_fb);
				}
			}
		}

		lm_fb.sortValues();
		lm_fb = lm_fb.subVector(fb_word_size);
		lm_fb.normalize();
		return lm_fb;
	}

	public double getDirichletPrior() {
		return prior_dir;
	}

	public int getFbDocSize() {
		return fb_doc_size;
	}

	public int getFbWordSize() {
		return fb_word_size;
	}

	public double getMixtureJM() {
		return mixture_jm;
	}

	public IntegerArrayMatrix getPassages(int dseq) throws Exception {
		IntegerArrayMatrix ret = new IntegerArrayMatrix();
		IntegerArrayMatrix doc = DocumentCollection.toMultiSentences(dc.get(dseq).getSecond());
		int len_d = doc.sizeOfEntries();

		IntegerArray d = new IntegerArray(len_d);
		for (IntegerArray sent : doc) {
			for (int w : sent) {
				d.add(w);
			}
		}

		int i = 0;

		while (true) {
			int start = i;
			int end = Math.min(start + len_psg_fix, len_d);
			IntegerArray psg = new IntegerArray(end - start);

			for (int j = start; j < end; j++) {
				psg.add(d.get(j));
			}
			ret.add(psg);

			i = end - window_size_psg;

			if (end == len_d) {
				break;
			}

		}

		return ret;
	}

	public SparseMatrix getPassageVectors(int docseq) throws Exception {
		IntegerArrayMatrix psgs = getPassages(docseq);
		List<SparseVector> rows = Generics.newArrayList(psgs.size());
		List<Integer> rowIdxs = Generics.newArrayList();
		for (IntegerArray psg : psgs) {
			rowIdxs.add(rows.size());

			rows.add(VectorUtils.toSparseVector(psg));
		}
		return new SparseMatrix(rowIdxs, rows);
	}

	public DenseVector getQueryBackgroundModel() {
		return lm_qbg;
	}

	private SparseVector getWordCounts(SparseMatrix docVecs) {
		return VectorUtils.toSparseVector(VectorUtils.toCounter(docVecs));
	}

	public void setDirichletPrior(double dirichlet_prior) {
		this.prior_dir = dirichlet_prior;
		dpe.setDirichletPrior(dirichlet_prior);
	}

	// public IntegerArrayMatrix getPassages(int docseq) throws Exception {
	// IntegerArrayMatrix ret = new IntegerArrayMatrix();
	// IntegerArrayMatrix doc = ds.getDocument(docseq).getSecond();
	// int len_d = doc.sizeOfEntries();
	// int loc = 0;
	// IntegerArray psg = new IntegerArray(len_psg_fix);
	//
	// for (IntegerArray sent : doc) {
	// for (int w : sent) {
	// psg.add(w);
	// loc++;
	// if (psg.size() == len_psg_fix || loc == len_d) {
	// ret.add(psg);
	// psg = new IntegerArray(len_psg_fix);
	// }
	// }
	// }
	//
	// return ret;
	// }

	public void setFbDocSize(int fb_doc_size) {
		this.fb_doc_size = fb_doc_size;
	}

	public void setFbWordSize(int fb_word_size) {
		this.fb_word_size = fb_word_size;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
		dpe.setMixtureJM(mixture_jm);
	}

	public void setPrintLog(boolean print_log) {
		this.print_log = print_log;
	}

	public void setQueryBackgroundModel(DenseVector lm_qbg) {
		this.lm_qbg = lm_qbg;
	}

	public void setUseDocumentPrior(boolean use_doc_prior) {
		this.use_doc_prior = use_doc_prior;
	}

	public SparseVector updateQueryModel(SparseVector lm_q, SparseVector lm_fb) {
		return VectorMath.addAfterMultiply(lm_q, 1 - mixture_fb, lm_fb, mixture_fb);
	}

}
