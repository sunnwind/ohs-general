package ohs.utils;

import java.util.Collections;
import java.util.List;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.ListMap;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics.ListType;

public class DataSplitter {

	public static IntegerMatrix group(IntegerArray x) {
		ListMap<Integer, Integer> lm = Generics.newListMap(ListType.LINKED_LIST);
		for (int i = 0; i < x.size(); i++) {
			lm.put(x.get(i), i);
		}

		IntegerArray ls = new IntegerArray(lm.keySet());
		ls.sort(false);

		IntegerMatrix ret = new IntegerMatrix(ls.size());

		for (int l : ls) {
			ret.add(new IntegerArray(lm.get(l)));
		}
		return ret;
	}

	public static IntegerMatrix split(IntegerArray x, double[] props) {
		double[] foldMaxIdxs = ArrayUtils.copy(props);
		int fold_size = foldMaxIdxs.length;

		ArrayMath.cumulate(foldMaxIdxs, foldMaxIdxs);

		IntegerMatrix ret = new IntegerMatrix(fold_size);
		ret.ensure(fold_size - 1);

		for (int i = 0; i < fold_size; i++) {
			foldMaxIdxs[i] = Math.rint(foldMaxIdxs[i] * x.size());
		}

		for (int i = 0, j = 0; i < x.size(); i++) {
			int max_idx = (int) foldMaxIdxs[j];
			if (i >= max_idx && j < fold_size) {
				j++;
			}
			ret.get(i).add(x.get(i));
		}
		return ret;
	}

	public static IntegerMatrix split(IntegerArray x, int group_size) {
		int group_cnt = (x.size() / group_size) + 1;
		IntegerMatrix ret = new IntegerMatrix(group_cnt);
		int i = 0;
		while (i < x.size()) {
			int j = Math.min(x.size(), i + group_size);
			ret.add(x.subArray(i, j));
			i += group_size;
		}
		return ret;
	}

	/**
	 * @param x
	 * @param cnts
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

	public static IntegerMatrix splitGroups(IntegerMatrix G, double[] props) {
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

	public static IntegerMatrix splitGroups(IntegerMatrix G, int[] cnts) {
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
