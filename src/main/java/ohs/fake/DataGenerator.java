package ohs.fake;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import ohs.corpus.type.DocumentCollection;
import ohs.math.ArrayMath;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.common.IntPair;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class DataGenerator {

	private LDocumentCollection C;

	public DataGenerator(LDocumentCollection C) {
		this.C = C;
	}

	public LDocumentCollection generate() {

		LDocumentCollection ret = new LDocumentCollection();

		for (LDocument d : C) {

			if (d.getAttrMap().get("cor_title") != null) {
				LDocument d2 = LDocument.newDocument(d.toString());
				d2.set(0, LDocument.newDocument(d.getAttrMap().get("cor_title")).get(0));
				d2.getAttrMap().put("label", "non-fake");
				ret.add(d2);
			}

			String label = d.getAttrMap().get("label");

			if (label.equals("non-fake")) {
				CounterMap<String, String> cm1 = Generics.newCounterMap();
				CounterMap<String, String> cm2 = Generics.newCounterMap();

				for (int i = 0; i < d.size(); i++) {
					LSentence s = d.get(i);

					for (LToken t : s) {
						String w = t.getString(0);
						String p = t.getString(1);

						if (i == 0) {
							cm1.incrementCount(p, w + " " + p, 1);
						} else {
							cm2.incrementCount(p, w + " " + p, 1);
						}
					}
				}

				// System.out.println(cm2.toString());
				// System.out.println();

				ListMap<String, Integer> posToLocs1 = Generics.newListMap();
				ListMap<String, Integer> posToLocs2 = Generics.newListMap();
				Map<Integer, IntPair> map = Generics.newHashMap();

				int sum = 0;
				for (int i = 0; i < d.size(); i++) {
					LSentence s = d.get(i);
					for (int j = 0; j < s.size(); j++) {
						LToken t = s.get(j);
						String w = t.getString(0);
						String p = t.getString(1);

						if (p.startsWith("NNG") || p.startsWith("NNP") || p.startsWith("SN")) {
							int idx = sum + j;

							if (i == 0) {
								posToLocs1.put(p, idx);
							} else {
								posToLocs2.put(p, idx);
							}
							map.put(idx, new IntPair(i, j));
						}
					}
					sum += s.size();
				}

				for (String p : posToLocs1.keySet()) {
					List<Integer> locs1 = posToLocs1.get(p);
					List<Integer> locs2 = posToLocs2.get(p, false);

					if (locs2 == null) {
						continue;
					}

					Collections.shuffle(locs1);
					Collections.shuffle(locs2);

					int size = Math.min(locs1.size(), locs2.size());

					for (int i = 0; i < size; i++) {
						int idx1 = locs1.get(i);
						int idx2 = locs2.get(i);

						IntPair p1 = map.get(idx1);
						IntPair p2 = map.get(idx2);

						LToken t1 = d.get(p1.getFirst()).get(p1.getSecond());
						LToken t2 = d.get(p2.getFirst()).get(p2.getSecond());

						if (t1.getString(0).equals(t2.getString(0))) {
							continue;
						}

						System.out.println(t1.toString());
						System.out.println(t2.toString());
						System.out.println();

						LDocument d2 = LDocument.newDocument(d.toString());
						d2.get(p1.getFirst()).set(p1.getSecond(), t2);
						d2.get(p2.getFirst()).set(p2.getSecond(), t1);
						d2.getAttrMap().put("label", "fake");

						ret.add(d2);
					}
				}
			}
		}

		return ret;

	}

}
