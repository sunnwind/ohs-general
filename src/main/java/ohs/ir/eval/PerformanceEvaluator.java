package ohs.ir.eval;

import java.util.List;

import ohs.io.TextFileReader;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;

public class PerformanceEvaluator {

	public static final int[] top_ns = { 10 };

	public static final int TOP_N = 10;

	public static final int TOP_N_MAP = 1000;

	public static Performance evalute(CounterMap<String, String> resData, CounterMap<String, String> relvData) {
		return evalute(resData, relvData, TOP_N);
	}

	public static Performance evalute(CounterMap<String, String> resData, CounterMap<String, String> relvData, int top_n) {
		CounterMap<MetricType, String> cm = Generics.newCounterMap();
		for (String qid : resData.keySet()) {
			Counter<String> scores = resData.getCounter(qid);
			Counter<String> rels = relvData.getCounter(qid);

			List<String> dids = scores.getSortedKeys();
			double relevant = Metrics.relevant(dids, scores.size(), rels);
			double relevant_in_judgements = Metrics.relevant(rels);
			double relevant_at_n = Metrics.relevant(dids, top_n, rels);
			double retrieved = scores.size();

			cm.setCount(MetricType.RETRIEVED, qid, retrieved);
			cm.setCount(MetricType.RELEVANT, qid, relevant_in_judgements);
			cm.setCount(MetricType.RELEVANT_IN_RET, qid, relevant);
			cm.setCount(MetricType.RELEVANT_AT, qid, relevant_at_n);

			double precision = Metrics.precision(dids, top_n, rels);
			double ap = Metrics.averagePrecision(dids, TOP_N_MAP, rels, true);
			double ndcg = Metrics.normalizedDiscountedCumulativeGain(dids, top_n, rels);

			cm.setCount(MetricType.P, qid, precision);
			cm.setCount(MetricType.AP, qid, ap);
			cm.setCount(MetricType.NDCG, qid, ndcg);
		}
		return new Performance(top_n, cm);
	}

	public static List<Performance> evalute(CounterMap<String, String> resData, CounterMap<String, String> relvData, int[] top_ns) {
		List<Performance> ret = Generics.newArrayList(top_ns.length);
		for (int top_n : top_ns) {
			ret.add(evalute(resData, relvData, top_n));
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.println("process ends.");

	}

	public static CounterMap<String, String> readSearchResults(String fileName) {
		CounterMap<String, String> ret = Generics.newCounterMap();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String[] parts = reader.next().split("\t");
			String qid = parts[0];
			String docid = parts[1];
			double score = Double.parseDouble(parts[2]);
			ret.incrementCount(qid, docid, score);
		}
		reader.close();
		return ret;
	}

}
