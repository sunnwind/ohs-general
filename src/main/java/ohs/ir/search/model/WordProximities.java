package ohs.ir.search.model;

import java.util.Map.Entry;

import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;

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

	public static void symmetric(CounterMap<Integer, Integer> cm) {
		System.out.println("get symmetric matrix");
		
		ListMap<Integer, Integer> lm = Generics.newListMap();

		for (int i : cm.keySet()) {
			lm.put(i, Generics.newArrayList(cm.getCounter(i).keySet()));
		}

		for (int k1 : lm.keySet()) {
			for (int k2 : lm.get(k1)) {
				if (k1 == k2) {
					continue;
				}
				double v = cm.getCount(k1, k2);
				cm.incrementCount(k2, k1, v);
			}
		}
	}

}
