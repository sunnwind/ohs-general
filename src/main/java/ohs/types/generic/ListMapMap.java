package ohs.types.generic;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.utils.Generics.ListType;

public class ListMapMap<K, V, F> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7366754514015096846L;

	protected Map<K, ListMap<V, F>> ents;

	public ListType lt;

	public ListMapMap() {
		this(0, ListType.ARRAY_LIST);
	}

	public ListMapMap(int size, ListType lt) {
		this.lt = lt;
		ents = new HashMap<K, ListMap<V, F>>(size);
	}

	public ListMapMap(ListType lt) {
		this(0, lt);
	}

	public void clear() {
		clear(true);
	}

	public void clear(boolean deep_clear) {
		if (deep_clear) {
			for (ListMap<V, F> lm : ents.values()) {
				lm.clear(deep_clear);
			}
		}
		ents.clear();
	}

	public boolean containsKey(K key1) {
		return ents.containsKey(key1);
	}

	protected ListMap<V, F> ensure(K key1) {
		ListMap<V, F> ret = ents.get(key1);
		if (ret == null) {
			ret = new ListMap<V, F>(lt);
			ents.put(key1, ret);
		}
		return ret;
	}

	protected List<F> ensure(K key1, V key2) {
		return ensure(key1).get(key2, true);
	}

	public ListMap<V, F> get(K key1) {
		return get(key1, true);
	}

	public ListMap<V, F> get(K key1, boolean ensure) {
		return ensure ? ensure(key1) : ents.get(key1);
	}

	public List<F> get(K key1, V key2, boolean ensure) {
		List<F> ret = null;
		if (ensure) {
			ret = ensure(key1, key2);
		} else {
			ListMap<V, F> temp = ents.get(key1);
			if (temp != null) {
				ret = temp.get(key2);
			}
		}
		return ret;
	}

	public Set<K> keySet() {
		return ents.keySet();
	}

	public void put(K key, ListMap<V, F> value) {
		ents.put(key, value);
	}

	public void put(K key1, V key2, F value) {
		ensure(key1, key2).add(value);
	}

	public void put(K key1, V key2, List<F> values) {
		ensure(key1, key2).addAll(values);
	}

	public int size() {
		return ents.size();
	}

	@Override
	public String toString() {
		return toString(30);
	}

	public String toString(int num_print_entries) {
		StringBuffer sb = new StringBuffer();
		int cnt = 0;
		int cnt2 = 0;
		int cnt3 = 0;

		List<K> keys1 = new ArrayList<K>(ents.keySet());

		for (int i = 0; i < keys1.size() && i < num_print_entries; i++) {
			ListMap<V, F> innerEntries = ents.get(keys1.get(i));

			List<V> keys2 = new ArrayList<V>(innerEntries.keySet());

			for (int j = 0; j < keys2.size() && j < num_print_entries; j++) {
				List<F> values = innerEntries.get(keys2.get(j), false);

				sb.append(keys1.get(i));
				sb.append(" -> " + keys2.get(j));
				for (int k = 0; k < values.size() && k < num_print_entries; k++) {
					if (k == 0) {
						sb.append(" ->");
					}
					sb.append(" " + values.get(k));
				}
				sb.append("\n");
			}
		}

		return sb.toString();
	}

	public long totalSize() {
		long ret = 0;
		for (K k : ents.keySet()) {
			ret += ents.get(k).sizeOfEntries();
		}
		return ret;
	}

}
