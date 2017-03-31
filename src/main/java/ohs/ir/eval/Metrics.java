package ohs.ir.eval;

import java.util.List;

import ohs.math.ArrayUtils;
import ohs.math.CommonMath;
import ohs.types.generic.Counter;
import ohs.utils.Generics;

public class Metrics {

	public static <E> double averagePrecision(List<E> docids, int n, Counter<E> docRels, boolean use_all_relevant) {
		double relevant_in_ret = 0;
		double relevance = 0;
		double ap = 0;
		double p = 0;
		int rank = 0;

		for (int i = 0; i < docids.size() && i < n; i++) {
			E docid = docids.get(i);
			relevance = docRels.getCount(docid);

			if (relevance > 0) {
				rank = i + 1;
				p = ++relevant_in_ret / rank;
				ap += p;
			}
		}

		double relevant = relevant_in_ret;

		if (use_all_relevant) {
			relevant = relevant(docRels);
		}

		if (relevant > 0) {
			ap /= relevant;
		}
		return ap;
	}

	public static <E> double binaryPreference(List<E> docids, Counter<E> docRels) {
		return binaryPreference(docids, docRels, 0);
	}

	/**
	 * 
	 * Buckley, C., & Voorhees, E. M. (2004). Retrieval evaluation with incomplete information. In Proceedings of the 27th annual
	 * international conference on Research and development in information retrieval - SIGIR ’04 (p. 25). New York, New York, USA: ACM
	 * Press. doi:10.1145/1008992.1009000
	 * 
	 * @param <E>
	 * 
	 * @param docids
	 * @param docRels
	 * @param pseudo_nonrelevant_size
	 * @return
	 */
	private static <E> double binaryPreference(List<E> docids, Counter<E> docRels, double pseudo_nonrelevant_size) {
		double relevant = 0;
		Counter<Integer> nonrels = Generics.newCounter();

		for (int r = 0; r < docids.size(); r++) {
			E docid = docids.get(r);
			double relevance = docRels.getCount(docid);
			if (relevance > 0) {
				nonrels.setCount(r, relevant);
			} else {
				relevant++;
			}
		}

		double ret = 0;
		for (int rank : nonrels.keySet()) {
			double nonrel_at_rank = nonrels.getCount(rank);
			ret += (1 - (nonrel_at_rank / (relevant + pseudo_nonrelevant_size)));
		}
		ret /= relevant;
		return ret;
	}

	public static <E> double binaryPreference10(List<E> docids, Counter<E> docRels) {
		return binaryPreference(docids, docRels, 10);
	}

	public static double[] discountedCumulativeGain(double[] gains) {
		double[] dcgs = new double[gains.length];
		double dcg = 0;

		for (int i = 0; i < gains.length; i++) {
			double rank = i + 1;
			double discount_factor = 1 / CommonMath.log2(1 + rank);
			double dg = discount_factor * gains[i];
			dcg += dg;
			dcgs[i] = dcg;
		}
		return dcgs;
	}

	public static double f1(double precision, double recall) {
		double ret = 0;
		if (precision > 0 || recall > 0) {
			ret = 2 * precision * recall / (precision + recall);
		}
		return ret;
	}

	public static <E> double f1(List<E> docids, int n, Counter<E> docRels) {
		double precision = precision(docids, n, docRels);
		double recall = recall(docids, n, docRels);
		return f1(precision, recall);
	}

	public static <E> double[] gain(List<E> docids, int n, Counter<E> docRels) {
		double[] gains = new double[n];
		for (int i = 0; i < n; i++) {
			E docid = docids.get(i);
			double relevance = docRels.getCount(docid);
			double gain = Math.pow(2, relevance) - 1;
			gains[i] = gain;
		}
		return gains;
	}

	public static double improvement(double old_value, double new_value) {
		double ret = 0;
		if (old_value != 0) {
			ret = (new_value - old_value) / old_value;
		}
		return ret;
	}

	/**
	 * Yilmaz, E., & Aslam, J. A. (2006). Estimating average precision with incomplete and imperfect judgments. In Proceedings of the 15th
	 * ACM international conference on Information and knowledge management - CIKM ’06 (p. 102). New York, New York, USA: ACM Press.
	 * doi:10.1145/1183614.1183633
	 * 
	 * @param docIds
	 * @param docRels
	 * @return
	 */
	public static double inferredAveragePrecision(List<String> docIds, Counter<String> docRels) {
		double ret = 0;

		return ret;
	}

	public static void main(String[] args) {

		{

			double[] gains = { 3, 2, 3, 0, 1, 2 };
			double[] max_gains = { 3, 2, 3, 0, 1, 2 };

			ArrayUtils.quickSort(max_gains);

			double[] dcgs = discountedCumulativeGain(gains);
			double[] mdcgs = discountedCumulativeGain(max_gains);

			System.exit(0);

		}

		{
			Counter<String> docScores = new Counter<String>();
			Counter<String> docRelevances = new Counter<String>();

			for (int i = 0; i < 20; i++) {
				String docId = String.format("doc_%d", i);
				docScores.setCount(docId, 20f / (i + 1));
			}

			docRelevances.setCount("doc_0", 1);
			docRelevances.setCount("doc_1", 1);
			docRelevances.setCount("doc_5", 1);
			docRelevances.setCount("doc_10", 1);
			docRelevances.setCount("doc_16", 1);

			List<String> docIds = docScores.getSortedKeys();

			rankBiasedPrecision(docIds, 20, docRelevances, 0.8);
		}
		{
			Counter<String> docScores = new Counter<String>();
			docScores.setCount("http://abc.go.com/", 10);
			docScores.setCount("http://www.abcteach.com/", 9);
			docScores.setCount("http://abcnews.go.com/sections/scitech/", 8);
			docScores.setCount("http://www.abc.net.au/", 7);
			docScores.setCount("http://abcnews.go.com/", 6);
			docScores.setCount("http://abc.org/", 5);

			Counter<String> docRelevances = new Counter<String>();
			docRelevances.setCount("http://abc.go.com/", 5);
			docRelevances.setCount("http://www.abcteach.com/", 2);
			docRelevances.setCount("http://abcnews.go.com/sections/scitech/", 4);
			docRelevances.setCount("http://www.abc.net.au/", 4);
			docRelevances.setCount("http://abcnews.go.com/", 4);
			docRelevances.setCount("http://abc.org/", 4);

			normalizedDiscountedCumulativeGain(docScores.getSortedKeys(), docScores.size(), docRelevances);
		}

		{
			Counter<String> docScores = new Counter<String>();
			docScores.setCount("http://abc.go.com/", 10);
			docScores.setCount("http://www.abcteach.com/", 9);
			docScores.setCount("http://abcnews.go.com/sections/scitech/", 8);
			docScores.setCount("http://www.abc.net.au/", 7);
			docScores.setCount("http://abcnews.go.com/", 6);
			docScores.setCount("http://abc.org/", 5);

			Counter<String> docRelevances = new Counter<String>();
			docRelevances.setCount("http://abc.go.com/", 5);
			docRelevances.setCount("http://www.abcteach.com/", 2);
			docRelevances.setCount("http://abcnews.go.com/sections/scitech/", 4);
			docRelevances.setCount("http://www.abc.net.au/", 4);
			docRelevances.setCount("http://abcnews.go.com/", 4);
			docRelevances.setCount("http://abc.org/", 4);

			normalizedDiscountedCumulativeGain(docScores.getSortedKeys(), docScores.size(), docRelevances);
		}

		{

			Counter<String> docScores = new Counter<String>();
			Counter<String> docRelevances = new Counter<String>();
			double[] rels = new double[] { 3, 2, 3, 0, 1, 2 };

			for (int i = 0; i < rels.length; i++) {
				docRelevances.setCount(i + "", rels[i]);
				docScores.setCount(i + "", rels.length - i);
			}

			normalizedDiscountedCumulativeGain(docScores.getSortedKeys(), docScores.size(), docRelevances);
		}

		{
			{
				Counter<String> docScores = new Counter<String>();
				Counter<String> docRels = new Counter<String>();

				for (int i = 0; i < 20; i++) {
					String docId = String.format("doc_%d", i);
					docScores.setCount(docId, 20f / (i + 1));
				}

				docRels.setCount("doc_0", 3);
				docRels.setCount("doc_1", 2);
				docRels.setCount("doc_4", 1);
				docRels.setCount("doc_10", 1);
				docRels.setCount("doc_16", 1);

				List<String> docIds = docScores.getSortedKeys();

				int[] ns = { 2, 5 };

				for (int i = 0; i < ns.length; i++) {
					int n = ns[i];
					double p = precision(docIds, n, docRels);
					double map = averagePrecision(docIds, n, docRels, true);
					double ndcg = normalizedDiscountedCumulativeGain(docIds, n, docRels);
					System.out.println(String.format("%d\t%s\t%s\t%s", n, p, map, ndcg));
				}

			}
		}

		// System.out.println();
	}

	public static <E> double normalizedDiscountedCumulativeGain(List<E> docids, int n, Counter<E> docRels) {
		n = docids.size() < n ? docids.size() : n;

		double[] gains = gain(docids, n, docRels);
		double[] gains_max = ArrayUtils.copy(gains);
		ArrayUtils.quickSort(gains_max);

		double[] dcgs = discountedCumulativeGain(gains);
		double[] dcgs_max = discountedCumulativeGain(gains_max);

		double[] ndcgs = new double[n];

		for (int i = 0; i < n; i++) {
			double dcg = dcgs[i];
			double dcg_max = dcgs_max[i];
			double ndcg = 0;

			if (dcg_max > 0) {
				ndcg = dcg / dcg_max;
			}
			ndcgs[i] = ndcg;
		}

		// System.out.println("[DOC_NDCG]");
		// System.out.println(doc_ndcg.toStringSortedByValues(true, true,
		// doc_ndcg.size()));

		double ndcg_at_n = ndcgs[n - 1];
		return ndcg_at_n;
	}

	public static <E> double precision(List<E> docids, int n, Counter<E> docRels) {
		return relevant(docids, n, docRels) / n;
	}

	/**
	 * "Rank-Biased Precision for Measurement of Retrieval Effectiveness" at ACM TOIS' 2008.
	 * 
	 * @param <E>
	 * 
	 * @param docIds
	 * @param n
	 * @param docRels
	 * @param p
	 * @return
	 */
	public static <E> double rankBiasedPrecision(List<E> docIds, int n, Counter<E> docRels, double p) {
		double ret = 0;

		for (int i = 0; i < docIds.size() && i < n; i++) {
			E docId = docIds.get(i);
			double relevance = docRels.getCount(docId);
			if (relevance > 1) {
				relevance = 1;
			}
			ret += relevance * Math.pow(p, i);
		}
		ret *= (1 - p);
		return ret;
	}

	public static <E> double recall(List<E> docids, int n, Counter<E> docRels) {
		double relevant = relevant(docRels);
		double relevant_at_n = relevant(docids, n, docRels);
		double ret = 0;
		if (relevant > 0) {
			ret = relevant_at_n / relevant;
		}
		return ret;
	}

	public static <E> double reciprocalRank(List<E> docids, int n, Counter<E> docRels) {
		double ret = 0;
		for (int i = 0; i < docids.size() && i < n; i++) {
			E docId = docids.get(i);
			if (docRels.getCount(docId) > 0) {
				double rank = i + 1;
				ret = 1f / rank;
				break;
			}
		}
		return ret;
	}

	public static <E> double relevant(Counter<E> docRels) {
		double ret = 0;
		for (double relevance : docRels.values()) {
			if (relevance > 0) {
				ret++;
			}
		}
		return ret;
	}

	public static <E> double relevant(List<E> docids, int n, Counter<E> docRels) {
		double ret = 0;
		for (int i = 0; i < docids.size() && i < n; i++) {
			if (docRels.getCount(docids.get(i)) > 0) {
				ret++;
			}
		}
		return ret;
	}

	public static <E> double[] riskRewardFunction(Counter<E> base, Counter<E> other) {
		double risk = 0;
		double reward = 0;

		for (E qId : base.keySet()) {
			double score1 = base.getCount(qId);
			double score2 = other.getCount(qId);
			risk += Math.max(0, score1 - score2);
			reward += Math.max(0, score2 - score1);
		}
		risk /= base.size();
		reward /= base.size();
		return new double[] { risk, reward };
	}

	/**
	 * Risk-Reward Tradeoff
	 * 
	 * 1. Wang, L., Bennett, P.N., Collins-Thompson, K.: Robust ranking models via risk-sensitive optimization. Proceedings of the 35th
	 * international ACM SIGIR conference on Research and development in information retrieval - SIGIR ’12. p. 761. ACM Press, New York, New
	 * York, USA (2012).
	 * 
	 * @param <E>
	 * 
	 * 
	 * @param base
	 * @param other
	 * @param alpha
	 * @return
	 */

	public static <E> double riskRewardTradeoff(Counter<E> base, Counter<E> other) {
		return riskRewardTradeoff(base, other, 5);
	}

	public static <E> double riskRewardTradeoff(Counter<E> base, Counter<E> other, double alpha) {
		double[] rr = riskRewardFunction(base, other);
		double risk = rr[0];
		double reward = rr[1];
		double ret = reward - (1 + alpha) * risk;
		return ret;
	}

	/**
	 * Collins-Thompson, K., & Callan, J. (2007). Estimation and use of uncertainty in pseudo-relevance feedback. In Proceedings of the 30th
	 * annual international ACM SIGIR conference on Research and development in information retrieval - SIGIR ’07 (p. 303). New York, New
	 * York, USA: ACM Press. http://doi.org/10.1145/1277741.1277795
	 * 
	 * @param oldScores
	 * @param newScores
	 * @return
	 */
	public static <E> double robustnessIndex(Counter<E> oldScores, Counter<E> newScores) {
		double pos_cnt = 0;
		double neg_cnt = 0;

		for (E qid : oldScores.keySet()) {
			double old_score = oldScores.getCount(qid);
			double new_score = newScores.getCount(qid);

			if (old_score < new_score) {
				pos_cnt++;
			} else if (old_score > new_score) {
				neg_cnt++;
			}
		}
		double ret = 0;

		if (oldScores.size() > 0) {
			ret = (pos_cnt - neg_cnt) / oldScores.size();
		}
		return ret;
	}

}
