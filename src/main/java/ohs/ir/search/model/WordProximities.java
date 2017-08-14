package ohs.ir.search.model;

import java.util.Map.Entry;

import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class WordProximities {

	public static CounterMap<Integer, Integer> hal(IntegerArray d, int window_size, boolean symmetric) {
		CounterMap<Integer, Integer> ret = Generics.newCounterMap();
		for (int i = 0; i < d.size(); i++) {
			int w1 = d.get(i);
			if (w1 < 0) {
				continue;
			}
			for (int j = i + 1; j < Math.min(d.size(), i + window_size); j++) {
				int w2 = d.get(j);
				if (w2 < 0) {
					continue;
				}
				double dist = j - i;
				double sim = 1d / dist;
				ret.incrementCount(w1, w2, sim);
				if (symmetric) {
					ret.incrementCount(w2, w1, sim);
				}
			}
		}
		return ret;
	}

	public static CounterMap<Integer, Integer> hal(IntegerArrayMatrix d, int window_size, boolean symmetric) {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		for (IntegerArray s : d) {
			cm.incrementAll(hal(s, window_size, symmetric));
		}
		return cm;
	}

	public static CounterMap<Integer, Integer> symmetric(CounterMap<Integer, Integer> cm) {
		CounterMap<Integer, Integer> ret = Generics.newCounterMap(cm);
		for (Entry<Integer, Counter<Integer>> e1 : cm.getEntrySet()) {
			int w1 = e1.getKey();
			for (Entry<Integer, Double> e2 : e1.getValue().entrySet()) {
				int w2 = e2.getKey();
				double v = e2.getValue();
				if (w1 != w2) {
					ret.incrementCount(w2, w1, v);
				}
			}
		}
		return ret;
	}

}
