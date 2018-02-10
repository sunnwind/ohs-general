package ohs.fake;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;

public class DataExpander {

	public static LDocumentCollection expand1(LDocumentCollection C) {
		LDocumentCollection C2 = new LDocumentCollection();

		ListMap<String, Integer> lm = Generics.newListMap();

		for (int i = 0; i < C.size(); i++) {
			LDocument d = C.get(i);
			String label = d.getAttrMap().get("label");
			lm.put(label, i);
		}

		List<Integer> l = lm.get("non-fake");

		for (int i = 0; i < l.size(); i++) {
			int target_loc = l.get(i);
			LDocument d1 = C.get(target_loc);
			LDocument d3 = d1.clone();

			int max_changes = 1;

			if (d1.size() > 20) {
				max_changes = 2;
			}

			int num_changes = ArrayMath.random(1, max_changes);
			Set<Integer> changedLocs = Generics.newHashSet();

			for (int j = 0; j < num_changes; j++) {

				LDocument d2 = null;

				{
					int sample_loc = -1;
					do {
						int k = ArrayMath.random(0, l.size());
						sample_loc = l.get(k);
					} while (target_loc == sample_loc);
					d2 = C.get(sample_loc);
				}

				int loc1 = -1;

				while (true) {
					loc1 = ArrayMath.random(1, d1.size() - 1);
					if (!changedLocs.contains(loc1)) {
						changedLocs.add(loc1);
						break;
					}
				}

				int loc2 = ArrayMath.random(1, d2.size() - 1);

				LSentence s1 = d1.get(loc1);
				LSentence s2 = d2.get(loc2);

				Indexer<String> idxer = Generics.newIndexer();

				SparseVector sv1 = VectorUtils.toSparseVector(s1.getCounter(0), idxer, true);
				SparseVector sv2 = VectorUtils.toSparseVector(s2.getCounter(0), idxer, true);
				double cosine = VectorMath.cosine(sv1, sv2);

				if (cosine < 0.5) {
					d3.set(loc1, d2.get(loc2).clone());
					d3.getAttrMap().put("label", "fake");
				}
			}

			C2.add(d3);
		}

		LDocumentCollection ret = new LDocumentCollection();

		for (int loc : lm.get("fake")) {
			ret.add(C.get(loc));
		}
		ret.addAll(C2);
		return ret;

	}

	public static LDocumentCollection expand2(LDocumentCollection C) {

		ListMap<String, Integer> lm = Generics.newListMap();

		for (int i = 0; i < C.size(); i++) {
			LDocument d = C.get(i);
			String label = d.getAttrMap().get("label");
			lm.put(label, i);
		}

		int num_fakes = lm.get("fake").size() * 3;

		List<Integer> locs = lm.get("non-fake");

		Collections.shuffle(locs);

		LDocumentCollection ret = new LDocumentCollection();

		for (int i = 0; i < locs.size(); i++) {
			int d_loc1 = locs.get(i);

			LDocument d1 = C.get(locs.get(i));

			if (i < num_fakes) {

				int d_loc2 = -1;
				LDocument d2 = null;

				do {
					d_loc2 = ArrayMath.random(0, C.size());
					d2 = C.get(d_loc2);
				} while (d2.size() < 5 && d_loc1 != d_loc2);

				int s_loc1 = ArrayMath.random(1, d1.size() - 1);
				int s_loc2 = ArrayMath.random(1, d2.size() - 1);

				LDocument d3 = d1.clone();

				LSentence s1 = d1.get(s_loc1);
				LSentence s2 = d2.get(s_loc2);

				d3.set(s_loc1, s2.clone());
				d3.getAttrMap().put("label", "fake");

				ret.add(d1);
				ret.add(d3);
			} else {
				ret.add(d1);
			}
		}

		for (int loc : lm.get("fake")) {
			ret.add(C.get(loc));
		}

		return ret;
	}

	public DataExpander() {

	}

}
