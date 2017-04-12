package ohs.utils;

import java.util.Collections;
import java.util.List;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.ListMap;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics.ListType;

public class DataSplitter {

	public static IntegerArrayMatrix group(IntegerArray x) {
		ListMap<Integer, Integer> lm = Generics.newListMap(ListType.LINKED_LIST);
		for (int i = 0; i < x.size(); i++) {
			lm.put(x.get(i), i);
		}

		IntegerArray ls = new IntegerArray(lm.keySet());
		ls.sort(false);

		IntegerArrayMatrix ret = new IntegerArrayMatrix(ls.size());

		for (int l : ls) {
			ret.add(new IntegerArray(lm.get(l)));
		}
		return ret;
	}

	public static IntegerArrayMatrix split(IntegerArray x, double[] props) {
		double[] foldMaxIdxs = ArrayUtils.copy(props);
		int fold_size = foldMaxIdxs.length;

		ArrayMath.cumulate(foldMaxIdxs, foldMaxIdxs);

		IntegerArrayMatrix ret = new IntegerArrayMatrix(fold_size);
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

	public static IntegerArrayMatrix split(IntegerArray x, int group_size) {
		int group_cnt = (x.size() / group_size) + 1;
		IntegerArrayMatrix ret = new IntegerArrayMatrix(group_cnt);
		int i = 0;
		while (i < x.size()) {
			int j = Math.min(x.size(), i + group_size);
			ret.add(x.subArray(i, j));
			i += group_size;
		}
		return ret;
	}

	public static IntegerArrayMatrix split(IntegerArray x, int[] cnts) {
		int fold_size = cnts.length;

		IntegerArrayMatrix ret = new IntegerArrayMatrix();
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

	public static IntegerArrayMatrix splitGroups(IntegerArrayMatrix G, double[] props) {
		int label_size = G.size();
		int fold_size = props.length;

		IntegerArrayMatrix ret = new IntegerArrayMatrix();
		ret.ensure(G.size() - 1);

		for (int i = 0; i < label_size; i++) {
			IntegerArray g = G.get(i);
			IntegerArrayMatrix S = split(g, props);

			for (int j = 0; j < fold_size; j++) {
				ret.get(j).addAll(S.get(j));
			}
		}

		return ret;
	}

	public static IntegerArrayMatrix splitGroups(IntegerArrayMatrix G, int[] cnts) {
		int label_size = G.size();
		int fold_size = cnts.length;

		IntegerArrayMatrix ret = new IntegerArrayMatrix();
		ret.ensure(fold_size - 1);

		for (int i = 0; i < label_size; i++) {
			IntegerArrayMatrix S = split(G.get(i), cnts);

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
