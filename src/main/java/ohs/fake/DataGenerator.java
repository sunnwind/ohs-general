package ohs.fake;

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

public class DataGenerator {

	private LDocumentCollection C;

	public DataGenerator(LDocumentCollection C) {
		this.C = C;
	}

	public LDocumentCollection generate() {
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
		ret.addAll(C);
		ret.addAll(C2);

		return ret;

	}

}
