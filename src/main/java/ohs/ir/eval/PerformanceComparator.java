package ohs.ir.eval;

import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;

public class PerformanceComparator {

	public static void compare(Performance base, Performance other) {
		CounterMap<MetricType, String> qbs = base.getQueryScores();
		CounterMap<MetricType, String> qos = other.getQueryScores();
		CounterMap<MetricType, String> qoimprs = other.getQueryImprovements();
		qoimprs.clear();

		for (MetricType mt : qbs.keySet()) {
			Counter<String> qb = qbs.getCounter(mt);
			Counter<String> qo = qos.getCounter(mt);

			for (String qid : qb.keySet()) {
				double impr = Metrics.improvement(qb.getCount(qid), qo.getCount(qid));
				qoimprs.setCount(mt, qid, impr);
			}
		}

		Counter<MetricType> b = base.getScores();
		Counter<MetricType> o = other.getScores();
		Counter<MetricType> imprs = other.getImprovements();
		imprs.clear();

		for (MetricType mt : b.keySet()) {
			Counter<String> qb = qbs.getCounter(mt);
			Counter<String> qo = qos.getCounter(mt);

			double ri = Metrics.robustnessIndex(qb, qo);
			double impr = Metrics.improvement(b.getCount(mt), o.getCount(mt));

			other.getRobustnessIndexes().setCount(mt, ri);
			other.getImprovements().setCount(mt, impr);
		}
	}

}
