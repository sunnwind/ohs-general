package ohs.utils;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.ListMap;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

public class DataSplitter {

	/**
	 * @param Y
	 *            data labels
	 * @return labels x data locations
	 */
	public static IntegerMatrix groupByLabels(IntegerArray Y) {
		ListMap<Integer, Integer> lm = Generics.newListMap();
		for (int i = 0; i < Y.size(); i++) {
			lm.put(Y.get(i), i);
		}

		IntegerArray L = new IntegerArray(lm.keySet());
		L.sort(false);

		IntegerMatrix ret = new IntegerMatrix(L.size());

		for (int l : L) {
			ret.add(new IntegerArray(lm.get(l)));
		}
		return ret;
	}

	public static IntegerMatrix split(IntegerArray L, double[] props) {
		double[] foldMaxIdxs = ArrayUtils.copy(props);
		int fold_size = foldMaxIdxs.length;

		ArrayMath.cumulate(foldMaxIdxs, foldMaxIdxs);

		IntegerMatrix ret = new IntegerMatrix(fold_size);
		ret.ensureCapacity(fold_size);

		for (int i = 0; i < fold_size; i++) {
			foldMaxIdxs[i] = Math.rint(foldMaxIdxs[i] * L.size());
			int size = (int) foldMaxIdxs[i];
			if (i > 0) {
				size -= foldMaxIdxs[i - 1];
			}
			ret.add(new IntegerArray(size));
		}

		for (int i = 0, j = 0; i < L.size(); i++) {
			int max_idx = (int) foldMaxIdxs[j];
			if (i >= max_idx && j < fold_size) {
				j++;
			}
			ret.get(j).add(L.get(i));
		}
		return ret;
	}

	public static IntegerMatrix split(IntegerArray L, int size) {
		int group_cnt = (L.size() / size) + 1;
		IntegerMatrix ret = new IntegerMatrix(group_cnt);
		int i = 0;
		while (i < L.size()) {
			int j = Math.min(L.size(), i + size);
			ret.add(L.subArray(i, j));
			i += size;
		}
		return ret;
	}

	/**
	 * @param x
	 *            data
	 * @param cnts
	 *            data sizes of groups
	 * 
	 * @return
	 */
	public static IntegerMatrix split(IntegerArray x, int[] cnts) {
		int fold_size = cnts.length;

		IntegerMatrix ret = new IntegerMatrix();
		ret.ensure(fold_size - 1);

		for (int i = 0, k = 0; i < fold_size; i++) {
			for (int j = 0; j < cnts[i] && k < x.size(); j++) {
				ret.add(i, x.get(k++));
			}
		}
		ret.trimToSize();
		return ret;
	}

	public static <E> List<E>[] split(List<E> ids, int num_folds) {
		return splitInOrder(ids, ArrayMath.array(num_folds, 1f / num_folds));
	}

	public static IntegerMatrix splitGroupsByLabels(IntegerArray L, double[] probs) {
		return splitGroupsByLabels(groupByLabels(L), probs);
	}

	public static IntegerMatrix splitGroupsByLabels(IntegerArray L, int[] cnts) {
		return splitGroupsByLabels(groupByLabels(L), cnts);
	}

	public static IntegerMatrix splitGroupsByLabels(IntegerMatrix G, double[] props) {
		int label_size = G.size();
		int fold_size = props.length;

		IntegerMatrix ret = new IntegerMatrix();
		ret.ensure(G.size() - 1);

		for (int i = 0; i < label_size; i++) {
			IntegerArray g = G.get(i);
			IntegerMatrix S = split(g, props);

			for (int j = 0; j < fold_size; j++) {
				ret.get(j).addAll(S.get(j));
			}
		}

		return ret;
	}

	/**
	 * @param G
	 *            labels x data locations
	 * @param cnts
	 *            data size for each label
	 * @return
	 */
	public static IntegerMatrix splitGroupsByLabels(IntegerMatrix G, int[] cnts) {
		int label_size = G.size();
		int fold_size = cnts.length;

		IntegerMatrix ret = new IntegerMatrix();
		ret.ensure(fold_size - 1);

		for (int i = 0; i < label_size; i++) {
			IntegerMatrix S = split(G.get(i), cnts);

			for (int j = 0; j < fold_size; j++) {
				ret.get(j).addAll(S.get(j));
			}
		}

		return ret;
	}

	public static <E> List<E>[] splitInOrder(List<E> locs, double[] proportions) {
		double[] foldMaxIdxs = ArrayUtils.copy(proportions);

		ArrayMath.cumulate(foldMaxIdxs, foldMaxIdxs);

		List<E>[] ret = new List[proportions.length];

		for (int i = 0; i < foldMaxIdxs.length; i++) {
			ret[i] = Generics.newArrayList();
			foldMaxIdxs[i] = Math.rint(foldMaxIdxs[i] * locs.size());
		}

		Collections.shuffle(locs);

		for (int i = 0, j = 0; i < locs.size(); i++) {
			int max_idx = (int) foldMaxIdxs[j];

			if (i >= max_idx && j < ret.length) {
				j++;
			}
			ret[j].add(locs.get(i));
		}
		return ret;
	}
}
